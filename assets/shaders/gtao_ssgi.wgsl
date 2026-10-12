// GTAO-based screen-space GI, with a per-direction fallback to the world
// radiance-cascade probes.
//
// One thread per pixel. For the pixel's surface point `p` (reconstructed from
// the deferred depth buffer) and its normal `n`, the hemisphere is split into
// `ss.slices` azimuth slices of width `2*pi/slices`. Each slice is a horizon
// search: march `ss.steps` world-space radii out along the slice's tangent
// direction, project each step back into the screen, and read the surface the
// depth buffer records there. A sample above the pixel's tangent plane is an
// occluder at elevation `theta`; the slice integral over elevation is
// `0.5 * dphi * d(sin^2 theta)` (since d(omega) = cos(theta) sin(theta) dtheta
// dphi and n.w = cos(theta)), so a newly-reached elevation band adds the
// occluder's directly-lit colour times that weight.
//
// A slice whose march leaves the screen, or reaches the radius without finding
// any occluder, is *off screen* for the band it does not cover: that band is
// filled with our own GI - the world cascade probe field read at `p`, its SH
// reconstructed as radiance in the band's direction (`sh_radiance`). That is
// the fallback: screen space where the depth buffer can answer, the cascades
// everywhere it cannot.
//
// Radiance on an on-screen hit is the current frame's directly-lit colour
// (this pass runs at EarlyPostProcess, after deferred lighting and before
// `shade`, so the view target holds direct light only - no history, no
// feedback). Output is a per-pixel irradiance the `shade` pass multiplies by
// albedo/pi; the alpha channel carries the on-screen coverage fraction for
// debugging.
//
// Cascade 0 stays in world space and is part of the fallback, so this pass is
// purely additive detail over the existing solution.

#import s_rc::heightmap_trace::{uniforms, oct_decode, sh_basis, SH_STRIDE, PROBES_PER_CELL, PI, is_solid, texel_at, probe_visible}

// Main camera bindings.
@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var gbuffer: texture_2d<u32>;
// The view target after deferred lighting: directly-lit HDR colour.
@group(1) @binding(2) var direct_color: texture_2d<f32>;
// The world cascade probe field (blend_sh's cross-fade), the fallback source.
@group(1) @binding(3) var<storage, read> sh_probes: array<vec4<f32>>;
// Per-pixel output: xyz = irradiance, w = on-screen coverage (debug).
@group(1) @binding(4) var<storage, read_write> gtao_out: array<vec4<f32>>;
@group(1) @binding(5) var<uniform> ss: GtaoUniforms;

struct GtaoUniforms {
    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    // Camera world position (unused for now; kept for the eye-relative bias).
    eye: vec3<f32>,
    // World radius each slice marches out to, m.
    radius_m: f32,
    // Screen size in pixels, and the fixed pixel-row stride of `gtao_out`
    // (so `shade` can index it without knowing the live size).
    screen: vec2<u32>,
    stride: u32,
    slices: u32,
    steps: u32,
    step_bias: f32,
    // Maximum screen-space search radius in pixels (clamps far-plane
    // projections of the world radius to prevent reaching foreground).
    max_screen_radius_px: u32,
    enabled: u32,
    pad0: u32,
}

const TAU: f32 = 6.28318530718;
const HALF_PI: f32 = 1.57079632679;
// Match `shade_cascades.wgsl`'s normal-biased fetch anchor (BIAS_NORMAL) when
// reading the world field for the fallback, so the fallback stands in for the
// same field the world gather would have read.
const WORLD_BIAS_NORMAL: f32 = 0.6;
// Thickness heuristic: reject occluders significantly in front of the ray's
// expected hit distance. Without this, foreground objects (which only have
// front-face depth) appear as infinitely thick monoliths occluding distant
// surfaces. If the occluder's distance along the ray is less than the step
// distance minus this thickness, it's a separate object.
const OCCLUDER_THICKNESS_M: f32 = 0.2;

fn hash12(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// World position of the visible surface at pixel `px`; w = 0 on background.
fn world_at(px: vec2<i32>) -> vec4<f32> {
    let d = textureLoad(depth_tex, px, 0);
    if d <= 0.0 {
        return vec4<f32>(0.0);
    }
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / vec2<f32>(ss.screen);
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let w = ss.world_from_clip * clip;
    return vec4<f32>(w.xyz / w.w, 1.0);
}

// The deferred G-buffer normal at pixel `px`.
fn normal_at(px: vec2<i32>) -> vec3<f32> {
    let g = textureLoad(gbuffer, px, 0);
    return oct_decode(vec2<f32>(f32(g.a & 0xFFFu), f32((g.a >> 12u) & 0xFFFu)) / 4095.0);
}

// Duff et al., "Building an Orthonormal Basis, Revisited". Columns are the
// tangent, bitangent and normal.
fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

// Radiance from direction `d` reconstructed from SH radiance coefficients
// (no cosine convolution - the coefficients already hold radiance).
fn sh_radiance(coefs: array<vec3<f32>, 9>, d: vec3<f32>) -> vec3<f32> {
    let b = sh_basis(d);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        e += coefs[i] * b[i];
    }
    return max(e, vec3<f32>(0.0));
}



@compute @workgroup_size(8, 8, 1)
fn gtao(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= ss.screen.x || gid.y >= ss.screen.y {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let idx = gid.y * ss.stride + gid.x;

    let pos = world_at(px);
    if pos.w < 0.5 {
        gtao_out[idx] = vec4<f32>(0.0);
        return;
    }
    let p = pos.xyz;
    let n = normal_at(px);
    let frame = tangent_frame(n);
    let tan_x = frame[0];
    let tan_y = frame[1];

    // Our GI at `p`, for the off-screen bands: the cascade-0 probe field, read
    // bilinearly in XZ and blended over the three layers by vertical distance.
    // The coefficients are kept so a band can be reconstructed in its own
    // direction.
    var wcoefs: array<vec3<f32>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        wcoefs[i] = vec3<f32>(0.0);
    }
    var wsum = 0.0;
    {
        let info = uniforms.cascades[0];
        // Read at the same anchor the world gather would (`p + n * BIAS_NORMAL`).
        let wp = p + n * WORLD_BIAS_NORMAL;
        let gu = (wp.xz - uniforms.hmap_min) / info.spacing - 0.5;
        let cell = vec2<i32>(floor(gu));
        let frac = gu - vec2<f32>(cell);
        for (var k = 0u; k < 4u; k = k + 1u) {
            let corner = vec2<u32>(k & 1u, k >> 1u);
            let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
            let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
            let w_corner = wxy.x * wxy.y;
            let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;
            var layer_w: array<f32, 3>;
            var layer_sum = 0.0;
            for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
                let c = sh_probes[(base_cell + layer) * SH_STRIDE + 9u].xyz;
                let dy = abs(c.y - wp.y);
                layer_w[layer] = 1.0 / (1e-3 + dy * dy);
                layer_sum += layer_w[layer];
            }
            if layer_sum <= 0.0 {
                continue;
            }
            for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
                let base = (base_cell + layer) * SH_STRIDE;
                let c = sh_probes[base + 9u].xyz;
                if probe_visible(p, n, c) {
                    let w = w_corner * (layer_w[layer] / layer_sum);
                    for (var i = 0u; i < 9u; i = i + 1u) {
                        wcoefs[i] += sh_probes[base + i].xyz * w;
                    }
                    wsum += w;
                }
            }
        }
    }
    let has_world = wsum > 1e-4;
    let sky = uniforms.sky_color.xyz;

    // World-space hash: stable slice rotation as camera moves (prevents ghosting).
    let rot = hash12(p.xz * 0.1) * TAU;

    let slices = max(ss.slices, 1u);
    let steps = max(ss.steps, 1u);
    let dphi = TAU / f32(slices);

    // Maximum screen-space radius clamp (prevents far pixels from
    // searching across the whole screen to foreground objects).
    let max_r_px = f32(ss.max_screen_radius_px);

    var e = vec3<f32>(0.0);
    var coverage = 0.0;
    for (var m = 0u; m < slices; m = m + 1u) {
        let phi = (f32(m) + 0.5) * dphi + rot;
        let tdir = normalize(tan_x * cos(phi) + tan_y * sin(phi));
        var prev_sin2 = 0.0;
        var last_dir = tdir;
        for (var k = 0u; k < steps; k = k + 1u) {
            let f = (f32(k) + 1.0) / f32(steps);
            // Denser near the pixel, where the slice's solid angle matters most.
            let r = ss.radius_m * f * f;
            let sp = p + tdir * r;
            let clip = ss.clip_from_world * vec4<f32>(sp, 1.0);
            if clip.w <= 0.0 {
                break;
            }
            let uv = (clip.xy / clip.w) * 0.5 + vec2<f32>(0.5);
            // Clamp screen-space search radius: compute projected delta from
            // the receiver pixel and skip if beyond max_r_px.
            let delta_px = (uv - vec2<f32>(px)) * vec2<f32>(ss.screen);
            if length(delta_px) > max_r_px {
                break;
            }
            if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
                break;
            }
            let hpx = vec2<i32>(uv * vec2<f32>(ss.screen));
            let sw = world_at(hpx);
            if sw.w < 0.5 {
                continue;
            }
            // Thickness check: if occluder is significantly in front of the
            // ray's expected hit distance, it's a separate foreground object.
            let ev = sw.xyz - p;
            let actual_dist = dot(ev, tdir);
            if r - actual_dist > OCCLUDER_THICKNESS_M {
                continue;
            }
            let h = dot(ev, n);
            let horiz = ev - n * h;
            let dh = length(horiz);
            if dh < 1e-4 {
                continue;
            }
            let theta = atan2(h, dh);
            if theta > 0.0 && theta < HALF_PI {
                let s2 = sin(theta) * sin(theta);
                if s2 > prev_sin2 {
                    let l = textureLoad(direct_color, hpx, 0).rgb;
                    e += l * (0.5 * (s2 - prev_sin2) * dphi);
                    prev_sin2 = s2;
                    last_dir = normalize(ev);
                }
            }
        }
        // The band the screen could not cover: our GI, evaluated in the band's
        // own direction. The band spans elevation asin(sqrt(prev_sin2)) radians
        // to the horizon; since the weight is uniform in sin^2 theta, evaluate
        // at the band's mean elevation, sin^2 = (prev_sin2 + 1) / 2.
        let rem = 0.5 * (1.0 - prev_sin2) * dphi;
        let s2_mid = 0.5 * (prev_sin2 + 1.0);
        let c_mid = sqrt(max(0.0, 1.0 - s2_mid));
        let band_dir = normalize(n * sqrt(s2_mid) + tdir * c_mid);
        var fallback = sky;
        if has_world {
            fallback = sh_radiance(wcoefs, band_dir);
        }
        e += fallback * rem;
        coverage += 0.5 * (1.0 - prev_sin2);
    }

    gtao_out[idx] = vec4<f32>(e, coverage / PI);
}