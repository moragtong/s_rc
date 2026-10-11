// Indirect diffuse from cascade 0's SH probes, added onto the lit HDR
// image. Two real probe layers per XZ cell (see `PROBES_PER_CELL`): a floor
// layer a constant distance above each column's stored floor surface and a
// ceiling layer the same distance below its stored ceiling. Because the
// probes sit *on* the surfaces they are about, the shade is a straight
// trilinear read - 4 XZ neighbours x 2 layers - with no parallax needed:
// there is no per-probe radius sphere to read against, and no vertical slide
// to correct for. The SH is evaluated in closed form against the surface
// normal (`sh_irradiance`), not the 32-sample parallax integral.
//
// `view_info.trilinear_enabled` (T) keeps the legacy single-layer bilinear
// reference - just the floor layer of the same probes, no vertical axis.
//
// `view_info.bilateral_enabled` (B) replaces the fixed footprint with a
// cross-bilateral (joint) filter over the probe field: range weights guided
// by the deferred G-buffer's pixel normal and depth and the probes' own
// baked normals and stored world positions. See the branch below.
//
// `view_info.normal_bias_enabled` (N) is the fetch mode toggle. Off: the probe
// grid is read at `p` and `probe_visible` rejects the probes behind the
// surface. On: the grid is read at `p + n * BIAS_NORMAL + V * BIAS_VIEW`
// instead - the cell in front of the surface - and `probe_visible` is skipped.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import s_rc::heightmap_trace::{uniforms, sh_basis, oct_decode, trace, MISS, PI, SH_STRIDE, height_agreement, PROBES_PER_CELL, probe_normal}

@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var gbuffer: texture_2d<u32>;
// The blended probe set: `blend_sh` writes mix(previous relight, current
// relight, alpha) here every frame before this pass runs, so the cross-fade
// costs one storage read rather than two. Probe positions and parallax radii
// are the same in both relights, so they survive the mix unchanged.
@group(1) @binding(2) var<storage, read> sh_probes: array<vec4<f32>>;

struct ShadeUniforms {
    view_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    indirect_enabled: u32,
    trilinear_enabled: u32,
    bilateral_enabled: u32,
    normal_bias_enabled: u32,
    bsp_recon_enabled: u32,
}
@group(1) @binding(3) var<uniform> view_info: ShadeUniforms;

// Cosine-distributed directions per probe for the irradiance integral.
const HEMISPHERE_SAMPLES: u32 = 32u;
const GOLDEN_ANGLE: f32 = 2.39996323;
// Push the shadow ray's origin off the shading point's own surface. A point
// lying on its column's floor or ceiling is inside that column's solid half
// space, so tracing from it unmodified reports a hit at t=0 and occludes every
// probe at once, including the ones actually in view.
const PROBE_BIAS: f32 = 0.02;
// Cross-bilateral filter kernels (see the `bilateral_enabled` branch in
// `shade`). The footprint is the pixel's 4 corner cells x 2 layers - the same
// 8 probes the trilinear read uses - because RC is noiseless: the probe
// field is already smooth, so there is nothing for a wide blur kernel to
// average and the range kernels only have to *choose among* the immediate
// neighbours. The normal kernel is a dot-product exponent; the plane kernel
// falls off over `BI_PLANE_M` metres of world-space separation from the
// pixel's tangent plane.
const BI_GAMMA: f32 = 32.0;
const BI_PLANE_M: f32 = 0.5;
// Weight normalization epsilon - raised to prevent dark bands at cell
// boundaries where probe taps get rejected (occlusion, normal rejection, etc.)
const WEIGHT_EPS: f32 = 1e-3;
// Normal-biased fetch (`view_info.normal_bias_enabled`, the N toggle). Instead
// of reading the probe grid at the shading point `p`, read it at
//
//     P_sample = p + n * BIAS_NORMAL + V * BIAS_VIEW,
//
// `n` the G-buffer normal and `V` the unit direction from the surface to the
// eye. Pushing the fetch along the normal moves it into the open half-space
// the surface faces, so the cell it lands in is the one *in front of* the
// surface and the probes behind it - through a wall, under a floor - are never
// selected in the first place. That is the job `probe_visible` does with a
// trace; with the bias on, the occlusion test is skipped (`shade` runs it only
// when the toggle is off), so the two are alternatives. BIAS_VIEW slides the
// fetch toward the eye as a secondary term; it defaults off. Both are metres,
// against cascade 0's 1 m spacing. BIAS_NORMAL starts at the 0.4 m guard band
// `mytodo.md` section 2 asks for (a probe closer than that to the surface it
// lights stops representing a field); it is the knob to tune for leakage.
const BIAS_NORMAL: f32 = 0.6;
const BIAS_VIEW: f32 = 0.0;
// Half-extent of the room's inner faces, from `house_gltf.py`: ROOM is 20x6x20
// with WALL = 0.3, so the interior runs -9.7..9.7 in both x and z.
//
// Tried as a clamp on the probe grid, then taken out in favour of rejecting on
// the winding signal: three quarters of the grid is the void outside this
// building, those probes sit in open columns so `height_agreement` does not
// stop them, and of the weight `probe_visible` would have
// rejected 90% was probes at |x| or |z| >= 9.7 - on a wall pixel, whose
// footprint is probes at 9.5 and 10.5, that is half of its footprint.
//
// Measured, clamp on vs the build without it (interior camera, pinned sun):
// mean |d| 0.0053, max 0.188, 1.15% of pixels past 0.05 and 0.23% past 0.10,
// every changed pixel darker and all of them in rows 58..533 x cols 407..829 -
// the wall the camera faces, losing the sky-lit radiance it was reading from
// outside. Inside the room it looked more correct; the exterior, lit from the
// interior, is expected to leak either way.
//
// To restore:
//   const PROBE_CLAMP_HALF: f32 = 9.7;
// and in the loop below:
//   if abs(slot.x) >= PROBE_CLAMP_HALF || abs(slot.z) >= PROBE_CLAMP_HALF { w = 0.0; }
const TAU: f32 = 6.28318530718;
// Per-pixel hash: breaks the 1m probe grid and the 32-sample hemisphere into
// noise instead of flat blocks. Seeded by the fragment coordinate (and the
// probe index below), so it is stable across frames and a screenshot stays
// sharp; it only decorrelates neighbouring pixels.
fn hash12(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Whether a probe is reachable from `p`. The heightmap records one floor and
// one ceiling per column, so it cannot represent a wall: without this test a
// probe on the far side of a wall, or a ground-floor probe under a second
// floor, bleeds straight through. `trace` returns MISS for a segment that
// stays in open space over its whole length.
fn probe_visible(p: vec3<f32>, n: vec3<f32>, center: vec3<f32>) -> bool {
    let to_probe = center - p;
    let dist = length(to_probe);
    if dist <= 1e-4 {
        return true;
    }
    let o = p + n * PROBE_BIAS;
    let t_end = max(dist - PROBE_BIAS, PROBE_BIAS);
    let hit = trace(o, to_probe / dist, PROBE_BIAS, t_end);
    return hit.t == MISS;

    // It was off for a while; when the build that turned it on (interior
    // camera, pinned sun, mean luminance 0.2040 without) was A/B:
    //
    //   occlusion on, is_solid = normals || winding : mean|d| 0.0362, 31.9% of
    //     px past 0.05, 4.01% brighter / 9.86% darker, luminance 0.2044
    //   occlusion on, is_solid = winding only       : mean|d| 0.0509, 64.7%
    //     px past 0.05, 33.91% brighter / 0.06% darker, luminance 0.2417
    //
    // The rule change is the bigger half of that, not the occlusion, and it
    // goes bright rather than dark - 0.0593 mean|d| between the two occluded
    // builds, max 0.92 - because `is_solid` is shared with `trace`, so it also
    // moves where the cascade rays stop and what the probes gather. Turning
    // occlusion on on top of it was worth 0.0362, and left 10% of pixels
    // darker, which is the wall rejection doing its job.
}

// The SH coefficients are radiance, and `sh_probes` is already the cross-fade
// between two relights (see blend_sh.wgsl), so this is a plain read: the blend
// is done once over 18k words instead of once per coefficient per pixel.
fn sh_coeff(base: u32, i: u32) -> vec3<f32> {
    return sh_probes[base + i].xyz;
}

// The 9 SH coefficients of a probe. `base` is invariant across all
// HEMISPHERE_SAMPLES samples, so they are loaded once per probe per pixel and
// the sample loop reads registers instead of re-reading `sh_probes` ~288
// times per probe per pixel (9 words x 32 samples). Measured against the
// in-loop read: bit-identical output, 13 -> 19 FPS on the interior camera -
// the shade pass's cost was redundant storage traffic, not the per-sample
// math or the probe_visible trace (removing either changed nothing).
fn sh_radiance(coefs: array<vec3<f32>, 9>, d: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(d);
    var l = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        l += coefs[i] * basis[i];
    }
    return max(l, vec3<f32>(0.0));
}

// Mix-then-clamp order for the gather: accumulate the 4 weighted coefficient
// vectors per layer and evaluate the clamped cosine lobe ONCE on the blended
// field, instead of clamping each probe's irradiance and then blending the
// scalars. The two orders are identical wherever the probe irradiances are
// already nonneg (evaluation is linear in the coefficients); they differ only
// in what happens to SH ringing along `n`, where per-probe clamping zeroes a
// negative lobe before it can cancel a positively-correlated neighbour. Mixed
// coefficients are the physically faithful reading of the interpolated field:
// irradiance of L(x) = sum w_k L_k is max(dot(L(x), b(n)), 0).
const SH_MIX_COEFFS: bool = true;

// Irradiance from a blended coefficient vector. Same kernel as
// `sh_irradiance(base, n)` but takes the already-mixed coefficients, so the
// clamp sees the field's value, not a per-probe value.
fn sh_irradiance_v(coefs: array<vec3<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 3>(PI, 2.094395, 0.785398);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let l = select(select(2u, 1u, i < 4u), 0u, i == 0u);
        e += coefs[i] * (basis[i] * band[l]);
    }
    return max(e, vec3<f32>(0.0));
}

// Irradiance from the SH convolved with the clamped cosine (no parallax).
// Still the read for the legacy single-layer branch and the bilateral gather.
fn sh_irradiance(base: u32, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 3>(PI, 2.094395, 0.785398);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let l = select(select(2u, 1u, i < 4u), 0u, i == 0u);
        e += sh_coeff(base, i) * (basis[i] * band[l]);
    }
    return max(e, vec3<f32>(0.0));
}

// Uniform cubic B-spline basis weights at fractional `t` in [0,1) for the four
// control points at offsets -1, 0, +1, +2 relative to the cell. B(x) is the
// standard cubic B-spline (support 2): the returned weights are
// B(t+1), B(t), B(1-t), B(2-t), which sum to 1 and are C^2 in `t`. This is the
// reconstruction half of the C^2 scheme whose control coefficients the
// `blend_sh` B-spline pre-filter supplies.
fn bspline_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    return array<f32, 4>(
        omt3 / 6.0,                          // B(t+1)
        (2.0 / 3.0) - t2 + 0.5 * t3,         // B(t)
        (2.0 / 3.0) - omt2 + 0.5 * omt3,     // B(1-t)
        t3 / 6.0,                            // B(2-t)
    );
}

// Duff et al., "Building an Orthonormal Basis, Revisited".
fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

// PREVIOUS (dead since the two-layer model): a 32-sample Monte-Carlo integral
// with parallax. The integral is redundant once the cosine lobe is pre-convolved
// into the coefficients - it only existed to vary the read direction per sample,
// which was the parallax correction: `d` was where the shading ray met the
// probe's farthest-hit sphere, not the normal. The two probe layers removed the
// need for it: every probe sits on the surface it is about, so the closed-form
// `sh_irradiance` below (no per-sample loop, no radius sphere) is the read.
// Kept, commented, for A/B - the measurement that retired it was the two-layer
// parity check (trilinear 4XZ x 2 layers vs the old virtual-slice trilinear).
// fn probe_irradiance(probe: u32, p: vec3<f32>, n: vec3<f32>, pixel: vec2<i32>, center: vec3<f32>) -> vec3<f32> {
//     let base = probe * SH_STRIDE;
//     let oc = p - center;
//     let radius = sh_probes[base + 9u].w;
//     let frame = tangent_frame(n);
//     var e = vec3<f32>(0.0);
//     var coefs: array<vec3<f32>, 9>;
//     for (var c = 0u; c < 9u; c++) {
//         coefs[c] = sh_probes[base + c].xyz;
//     }
//     let rot = hash12(vec2<f32>(pixel) * 0.37 + vec2<f32>(f32(probe & 255u) * 0.17, f32((probe >> 8u) & 255u) * 0.23)) * TAU;
//     for (var k = 0u; k < HEMISPHERE_SAMPLES; k++) {
//         let u = (f32(k) + 0.5) / f32(HEMISPHERE_SAMPLES);
//         let r = sqrt(u);
//         let phi = f32(k) * GOLDEN_ANGLE + rot;
//         let w = frame * vec3<f32>(r * cos(phi), r * sin(phi), sqrt(1.0 - u));
//         var d = w;
//         if radius > 0.0 {
//             let b = dot(oc, w);
//             let disc = b * b - (dot(oc, oc) - radius * radius);
//             if disc >= 0.0 {
//                 let t = -b + sqrt(disc);
//                 if t > 0.0 {
//                     d = normalize(oc + w * t);
//                 }
//             }
//         }
//         e += sh_radiance(coefs, d);
//     }
//     return e * (PI / f32(HEMISPHERE_SAMPLES));
// }

// ACTIVE: the closed-form cosine lobe, no integral and no parallax. The SH
// radiance coefficients are convolved by the clamped-cosine lobe: each band k
// scales by a fixed constant (`sh_irradiance`). The thread is the surface
// normal `n`, which is right now that every probe sits on the surface it
// reports about.

@fragment
fn shade(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    // Reverse Z: 0 is the far plane (sky).
    let depth = textureLoad(depth_tex, pixel, 0);
    let ro = (view_info.world_from_view * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    var p = vec3<f32>(0.0);
    var max_t = 1.0e9;
    if depth > 0.0 {
        let clip = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, depth, 1.0);
        let view = view_info.view_from_clip * clip;
        p = (view_info.world_from_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz;
        max_t = distance(p, ro);
    }

    if depth <= 0.0 || view_info.indirect_enabled == 0u {
        return vec4<f32>(0.0);
    }

    // Bevy's deferred G-buffer: r = sRGB base color + roughness, b =
    // reflectance, metallic, occlusion, a = 24-bit octahedral normal + flags.
    let g = textureLoad(gbuffer, pixel, 0);
    let base_color = pow(unpack4x8unorm(g.r).rgb, vec3<f32>(2.2));
    let metallic = unpack4x8unorm(g.b).g;
    let n = oct_decode(vec2<f32>(f32(g.a & 0xFFFu), f32((g.a >> 12u) & 0xFFFu)) / 4095.0);

    let info = uniforms.cascades[0];
    // Probe-grid read, currently unjittered. The original form jitters by up
    // to half a cell per axis, keyed per pixel: the bilinear tent then splits
    // its sample between neighbouring cells a pixel at a time, turning the 1m
    // probe structure - weight jumps, height-agreement shelf edges, relocated
    // probe discontinuities - into fine noise instead of block boundaries.
    //
    // It is off because the noise is screen-locked (`hash12(pixel)`), so it
    // stays in one place on the display while the scene moves under it and
    // reads as a shadow that follows the camera. Measured on the interior
    // camera with a frozen sun: the jittered indirect delta has high-frequency
    // energy 0.66 against 0.42 unjittered, and a worst 16px tile gradient of
    // 27.0 against 8.1 - the direct-only frame is 6.3, so unjittered is back at
    // the level of the underlying image. Mean indirect is unchanged (9.44 vs
    // 9.52), so it was decorrelating structure, not carrying light.
    //
    // To restore: `let dij = (vec2<f32>(hash12(vec2<f32>(pixel)), hash12(vec2<f32>(pixel) + 31.7)) - 0.5) / 2.0;`
    // It is worth trying again only if it is keyed to something world-relative
    // or animated per frame, so it dithers rather than sticking to the screen.
    let dij = vec2<f32>(0.0);
    // The fetch anchor. With the normal bias off it is the shading point
    // itself; with it on it is `p` pushed along the surface normal (and, if
    // BIAS_VIEW is set, toward the eye), so the cell selected is the one in
    // front of the surface. `probe_visible` is the other half of the toggle:
    // it is only run when the bias is off (see the tap tests below).
    var fetch = p;
    if view_info.normal_bias_enabled != 0u {
        let view_dir = (ro - p) / max(max_t, 1e-4);
        fetch = p + n * BIAS_NORMAL + view_dir * BIAS_VIEW;
    }
    // The 4 XZ neighbours of the fetch anchor, shared by both gather shapes.
    // Trilinear splits each bilinear tap on the vertical axis (see the branch
    // below); the bilinear gather reads them at their stored height.
    let gu = (fetch.xz - uniforms.hmap_min) / info.spacing - 0.5 + dij;
    let cell = vec2<i32>(floor(gu));
    // Raw in-cell fraction for the cubic B-spline reconstruction; the bilinear
    // shapes smooth it (below) so their tent hits zero slope at the cell edge.
    let ft = gu - vec2<f32>(cell);
    var frac = gu - vec2<f32>(cell);
    frac = frac * frac * frac * (frac * (frac * 6.0 - 15.0) + 10.0);
    var e = vec3<f32>(0.0);
    if view_info.bilateral_enabled != 0u {
        // Cross-bilateral (joint) filter over the immediate neighbours, guided
        // by the deferred G-buffer. RC is noiseless, so the field between a
        // pixel's 4 corner cells is already a linear interpolation of them:
        // a wide footprint would only re-smooth what the gather already made
        // smooth. The filter keeps the trilinear footprint (4 corner cells x
        // 2 layers = 8 probes) and replaces the fixed bilinear weights and
        // `height_agreement` fade with guide-driven weights:
        //
        //   spatial: the bilinear tent of the pixel's position in its cell,
        //   normal : max(0, n_p . n_q)^gamma, where n_p is the pixel's
        //            G-buffer normal and n_q is the *probe's* own normal.
        //            Every probe's normal is assigned by its bake: the
        //            heightmap G-buffer normal of its column (ceiling layer
        //            -> ceiling_normal, floor layer -> floor_normal), or,
        //            when the probe sat inside a wall and was moved sideways
        //            out to the nearest open column, the direction of that
        //            move - the way it faces out of the wall. On a wall the
        //            hard gate is replaced by a soft wrap (see Bug 4 in the
        //            loop below), otherwise the vertical floor/ceiling probe
        //            normals all score 0 and the wall weight collapses.
        //   plane  : exp(-|n_p . (X_q - X_p)| / BI_PLANE_M), the world-space
        //            distance from the probe to the pixel's tangent plane.
        //
        // Both depths enter in world space: X_p is the pixel's position
        // reconstructed from the prepass depth above (`p`), X_q is the probe
        // position the heightmap bake stored alongside its SH. The range
        // kernels are what keep edges sharp: a probe behind a corner or a
        // metre off the pixel's plane drops to ~zero weight, so the pixel
        // never blends across the discontinuity - and the floor and ceiling
        // layers separate by *normal*, not by a vertical fade. Open-column
        // fallback probes (no stored surface) already carry their layer's
        // vertical normal from the bake, and the plane weight sorts them out.
        var e_bi = vec3<f32>(0.0);
        var w_bi = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let corner = vec2<u32>(k & 1u, k >> 1u);
            let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
            let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
            let w_spatial = wxy.x * wxy.y;
            let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;
            for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
                let base = (base_cell + layer) * SH_STRIDE;
                let center = sh_probes[base + 9u].xyz;
                // The probe's own normal, as assigned by its bake: the
                // heightmap normal of its column, or the nudge direction if
                // the probe sat inside a wall and was moved sideways.
                let nq = probe_normal(0u, q, layer);
                let ndot = dot(n, nq);
                // Bug 4: a vertical wall's normal is orthogonal to the floor
                // and ceiling probe normals, so the hard orientation gate
                // `pow(max(0, ndot), BI_GAMMA)` returns 0 for every standard
                // probe and the wall's weight collapses onto the handful of
                // nudged probes whose horizontal normal happens to align. On a
                // wall, wrap the gate instead of clamping: accept vertical
                // probe normals with a soft ramp (`ndot = 0` -> 0.5). Every
                // other surface keeps the hard gate that preserves edges.
                // PREVIOUS (collapsed on walls): normal_weight = pow(max(0.0,
                // ndot), BI_GAMMA).
                let normal_weight = select(
                    pow(max(0.0, ndot), BI_GAMMA),
                    saturate(ndot * 0.5 + 0.5),
                    abs(n.y) < 0.5 && abs(nq.y) > 0.8,
                );
                let w = w_spatial
                    * normal_weight
                    * exp(-abs(dot(n, center - p)) / BI_PLANE_M);
                // The occlusion trace only runs for taps that survive the
                // kernels - the ones the guides already agree on - and only in
                // the `probe_visible` mode: the normal-biased fetch already
                // selects the cell in front of the surface, so it does not need
                // the trace.
                if w > WEIGHT_EPS
                    && (view_info.normal_bias_enabled != 0u || probe_visible(p, n, center))
                {
                    e_bi += w * sh_irradiance(base, n);
                    w_bi += w;
                }
            }
        }
        e = select(e_bi, e_bi / w_bi, w_bi > WEIGHT_EPS);
        return vec4<f32>(base_color / PI * e, 0.0);
    } else if view_info.bsp_recon_enabled != 0u {
        // Cubic B-spline reconstruction (`C` key / `S_RC_BSPLINE_RECON=1`).
        // The bilinear/trilinear read reconstructs with a 2x2 tent per layer,
        // which is only C^0: the field is continuous across a cell boundary but
        // its gradient kinks, so the probe grid reads as vertical/horizontal
        // seams under steep contrast. Here the in-cell fraction `ft` weights the
        // four control points per axis (offsets -1..+2) by the cubic B-spline
        // basis (`bspline_weights`), a 4x4 read per layer, which is C^2. Paired
        // with the `blend_sh` B-spline pre-filter (which supplies the control
        // coefficients) this is the full two-step scheme; the pre-filter on its
        // own cannot remove the kink because a pre-filter does not change the
        // reconstruction.
        //
        // Three real probe layers: floor (0), mid-air (1), ceiling (2).
        // Each layer is reconstructed from its own 4x4 patch and the three are
        // faded by the B-spline-blended layer heights.
        let wx = bspline_weights(ft.x);
        let wz = bspline_weights(ft.y);
        var cf_floor: array<vec3<f32>, 9>;
        var cf_mid: array<vec3<f32>, 9>;
        var cf_ceil: array<vec3<f32>, 9>;
        var e_floor = vec3<f32>(0.0);
        var e_mid = vec3<f32>(0.0);
        var e_ceil = vec3<f32>(0.0);
        var w_floor = 0.0;
        var w_mid = 0.0;
        var w_ceil = 0.0;
        var h_floor = 0.0;
        var h_mid = 0.0;
        var h_ceil = 0.0;
        for (var j = 0u; j < 4u; j++) {
            for (var k = 0u; k < 4u; k++) {
                let q = vec2<u32>(clamp(
                    cell + vec2<i32>(i32(k) - 1, i32(j) - 1),
                    vec2<i32>(0),
                    vec2<i32>(info.size) - 1,
                ));
                let w = wx[k] * wz[j];
                let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;
                for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
                    let base = (base_cell + layer) * SH_STRIDE;
                    let center = sh_probes[base + 9u].xyz;
                    if (layer == 0u) {
                        h_floor += w * center.y;
                    } else if (layer == 1u) {
                        h_mid += w * center.y;
                    } else {
                        h_ceil += w * center.y;
                    }
                    if w > 0.0
                        && (view_info.normal_bias_enabled != 0u || probe_visible(p, n, center))
                    {
                        if SH_MIX_COEFFS {
                            for (var i = 0u; i < 9u; i++) {
                                let c = sh_coeff(base, i) * w;
                                if (layer == 0u) {
                                    cf_floor[i] += c;
                                } else if (layer == 1u) {
                                    cf_mid[i] += c;
                                } else {
                                    cf_ceil[i] += c;
                                }
                            }
                        } else {
                            let read = sh_irradiance(base, n);
                            if (layer == 0u) {
                                e_floor += w * read;
                            } else if (layer == 1u) {
                                e_mid += w * read;
                            } else {
                                e_ceil += w * read;
                            }
                        }
                        if (layer == 0u) {
                            w_floor += w;
                        } else if (layer == 1u) {
                            w_mid += w;
                        } else {
                            w_ceil += w;
                        }
                    }
                }
            }
        }
        if SH_MIX_COEFFS {
            e_floor = select(
                sh_irradiance_v(cf_floor, n),
                sh_irradiance_v(cf_floor, n) / w_floor,
                w_floor > WEIGHT_EPS,
            );
            e_mid = select(
                sh_irradiance_v(cf_mid, n),
                sh_irradiance_v(cf_mid, n) / w_mid,
                w_mid > WEIGHT_EPS,
            );
            e_ceil = select(
                sh_irradiance_v(cf_ceil, n),
                sh_irradiance_v(cf_ceil, n) / w_ceil,
                w_ceil > WEIGHT_EPS,
            );
        } else {
            e_floor = select(e_floor, e_floor / w_floor, w_floor > WEIGHT_EPS);
            e_mid = select(e_mid, e_mid / w_mid, w_mid > WEIGHT_EPS);
            e_ceil = select(e_ceil, e_ceil / w_ceil, w_ceil > WEIGHT_EPS);
        }
        // Trilinear vertical blend: floor -> mid -> ceiling
        var t_floor = saturate((fetch.y - h_floor) / max(h_mid - h_floor, 1e-4));
        t_floor = t_floor * t_floor * (3.0 - 2.0 * t_floor);
        var t_ceil = saturate((fetch.y - h_mid) / max(h_ceil - h_mid, 1e-4));
        t_ceil = t_ceil * t_ceil * (3.0 - 2.0 * t_ceil);
        e = mix(mix(e_floor, e_mid, t_floor), mix(e_mid, e_ceil, t_ceil), 0.5);
        return vec4<f32>(base_color / PI * e, 0.0);
    } else if view_info.trilinear_enabled != 0u {
        // Trilinear over three real probe layers: floor (0), mid-air (1), ceiling (2).
        // Each of the 4 XZ neighbours holds probes at three heights. Each layer
        // is gathered bilinearly in XZ, then the three are faded by where `p`
        // sits between the bilinearly interpolated heights.
        var cf_floor: array<vec3<f32>, 9>;
        var cf_mid: array<vec3<f32>, 9>;
        var cf_ceil: array<vec3<f32>, 9>;
        var e_floor = vec3<f32>(0.0);
        var e_mid = vec3<f32>(0.0);
        var e_ceil = vec3<f32>(0.0);
        var w_floor = 0.0;
        var w_mid = 0.0;
        var w_ceil = 0.0;
        var h_floor = 0.0;
        var h_mid = 0.0;
        var h_ceil = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let corner = vec2<u32>(k & 1u, k >> 1u);
            let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
            let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
            let w = wxy.x * wxy.y;
            let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;
            for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
                let base = (base_cell + layer) * SH_STRIDE;
                let center = sh_probes[base + 9u].xyz;
                // The three layer heights, bilinearly merged, define the
                // vertical axis of the trilinear read.
                if (layer == 0u) {
                    h_floor += w * center.y;
                } else if (layer == 1u) {
                    h_mid += w * center.y;
                } else {
                    h_ceil += w * center.y;
                }
                if w > 0.0
                    && (view_info.normal_bias_enabled != 0u || probe_visible(p, n, center))
                {
                    // PREVIOUS (scalar order, keep for the SH_MIX_COEFFS=0
                    // build): the per-probe irradiance was clamped by
                    // `sh_irradiance` before blending, so a negative SH lobe
                    // along `n` was zeroed in place.
                    if !SH_MIX_COEFFS {
                        let read = sh_irradiance(base, n);
                        if (layer == 0u) {
                            e_floor += w * read;
                        } else if (layer == 1u) {
                            e_mid += w * read;
                        } else {
                            e_ceil += w * read;
                        }
                    }
                    // Blend the 9 coefficient vectors by `w`; the clamped
                    // cosine kernel is applied once to the field (see the
                    // `SH_MIX_COEFFS` comment at the top).
                    if SH_MIX_COEFFS {
                        for (var i = 0u; i < 9u; i++) {
                            let c = sh_coeff(base, i) * w;
                            if (layer == 0u) {
                                cf_floor[i] += c;
                            } else if (layer == 1u) {
                                cf_mid[i] += c;
                            } else {
                                cf_ceil[i] += c;
                            }
                        }
                    }
                    if (layer == 0u) {
                        w_floor += w;
                    } else if (layer == 1u) {
                        w_mid += w;
                    } else {
                        w_ceil += w;
                    }
                }
            }
        }
        // Renormalise each layer over the probes that survived, so an occluded
        // probe hands its share to the visible ones instead of subtracting
        // light. Without this the footprint sums to less than 1 and every
        // occlusion reads as a straight loss of energy rather than a
        // redistribution. In the coefficient order the same division is
        // applied to the evaluated field, which is equivalent because the
        // kernel is linear.
        if SH_MIX_COEFFS {
            e_floor = select(
                sh_irradiance_v(cf_floor, n),
                sh_irradiance_v(cf_floor, n) / w_floor,
                w_floor > WEIGHT_EPS,
            );
            e_mid = select(
                sh_irradiance_v(cf_mid, n),
                sh_irradiance_v(cf_mid, n) / w_mid,
                w_mid > WEIGHT_EPS,
            );
            e_ceil = select(
                sh_irradiance_v(cf_ceil, n),
                sh_irradiance_v(cf_ceil, n) / w_ceil,
                w_ceil > WEIGHT_EPS,
            );
        } else {
            e_floor = select(e_floor, e_floor / w_floor, w_floor > WEIGHT_EPS);
            e_mid = select(e_mid, e_mid / w_mid, w_mid > WEIGHT_EPS);
            e_ceil = select(e_ceil, e_ceil / w_ceil, w_ceil > WEIGHT_EPS);
        }
        // Blend only the nearest 2 layers: floor+mid or mid+ceiling
        var t_mid = saturate((fetch.y - h_floor) / max(h_ceil - h_floor, 1e-4));
        var e_mid_ceil = mix(e_mid, e_ceil, t_mid);
        var e_floor_mid = mix(e_floor, e_mid, t_mid);
        // Use mid as pivot: below mid blends floor+mid, above mid blends mid+ceiling
        var t_pivot = saturate((fetch.y - h_mid) / max(h_ceil - h_floor, 1e-4));
        e = mix(e_floor_mid, e_mid_ceil, t_pivot);
        return vec4<f32>(base_color / PI * e, 0.0);
    } else {
        // Legacy single-layer reference, for A/B: just the floor layer of the
        // same probes, bilinear in XZ, no vertical axis. Selected by
        // `S_RC_TRILINEAR=0` / the T key.
        var w_sum = 0.0;
        var e = vec3<f32>(0.0);
        for (var k = 0u; k < 4u; k++) {
            let corner = vec2<u32>(k & 1u, k >> 1u);
            let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
            let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
            let w = wxy.x * wxy.y;
            // Read all 3 layers and sum them
            for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
                let base = (info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL + layer) * SH_STRIDE;
                let center = sh_probes[base + 9u].xyz;
                if w > 0.0 && probe_visible(p, n, center) {
                    e += w * sh_irradiance(base, n) * height_agreement(center.y - p.y, info.spacing);
                    w_sum += w;
                }
            }
        }
        e = select(e, e / w_sum, w_sum > WEIGHT_EPS);
    }
    // The material's metallic from the deferred G-buffer reads ~1 for this
    // asset even though the room GLB has no material (Bevy's default is 0),
    // so `(1.0 - metallic)` silently killed the whole pass. The bake is
    // purely diffuse, so keep it diffuse here too.
    return vec4<f32>(base_color / PI * e, 0.0);
}
