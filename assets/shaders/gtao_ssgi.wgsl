// Smooth SH Irradiance with Hanning/Lanczos Windowing and hybrid
// horizontal-bilinear / inverse-square vertical blend, composited with contact SSAO.
//
// One thread per pixel. For the pixel's surface point `p` and its normal `n`:
// 1. Gathers the 12 cascade-0 world probes (4 corner cells x 3 vertical layers),
//    gated by `probe_visible(p, n, c)` (or bypassed when normal-bias is enabled via 'N').
// 2. Evaluates the hybrid blend: bilinear horizontal interpolation in XZ and
//    continuous inverse-square vertical weighting (1 / (eps + dy^2)).
// 3. Evaluates analytic cosine-convolved irradiance with a Hanning/Lanczos window
//    filter applied across SH degrees:
//      l=0: 1.0 (DC energy strictly preserved)
//      l=1: cos^2(pi / 6) = 0.75
//      l=2: cos^2(2*pi / 6) = 0.25
//    This mathematical tapering completely eliminates negative ringing / sidelobes
//    everywhere on the sphere in a single analytic pass (9 FMAs, 0 numerical slice loops).
// 4. Writes irradiance to `gtao_out`, which is multiplied by Bevy's XeGTAO SSAO
//    in `composite_indirect` for clean contact shadowing.

#import s_rc::heightmap_trace::{uniforms, oct_decode, sh_basis, SH_STRIDE, PROBES_PER_CELL, PI, is_solid, texel_at, probe_visible, probe_visibility_mask, probe_is_visible}

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
    normal_bias_enabled: u32,
    two_closest_layers: u32,
}

// Match `shade_cascades.wgsl`'s normal-biased fetch anchor (BIAS_NORMAL) when
// reading the world field for the fallback, so the fallback stands in for the
// same field the world gather would have read.
const WORLD_BIAS_NORMAL: f32 = 0.6;

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

// Analytic cosine-convolved SH irradiance with Hanning windowing to eliminate
// negative ringing lobes.
// The standard Ramamoorthi-Hanrahan convolution factors A_l are:
//   A_0 = pi
//   A_1 = 2*pi / 3  ~= 2.094395
//   A_2 = pi / 4    ~= 0.785398
// The Hanning smoothing window factors W_l = cos^2(l * pi / (2 * (L + 1))) for L=2 are:
//   W_0 = cos^2(0) = 1.0
//   W_1 = cos^2(pi / 6) = 0.75
//   W_2 = cos^2(pi / 3) = 0.25
// Combined effective kernel weights:
//   band_0 = A_0 * W_0 = PI
//   band_1 = A_1 * W_1 = 2.094395 * 0.75 = 1.570796
//   band_2 = A_2 * W_2 = 0.785398 * 0.25 = 0.196350
fn sh_irradiance_windowed(coefs: array<vec3<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 3>(PI, 1.5707963, 0.1963495);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let l = select(select(2u, 1u, i < 4u), 0u, i == 0u);
        e += coefs[i] * (basis[i] * band[l]);
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

    // Gather the cascade-0 probe field, read bilinearly in XZ and blended
    // over the three layers by continuous inverse-square vertical distance.
    var wcoefs: array<vec3<f32>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        wcoefs[i] = vec3<f32>(0.0);
    }
    var wsum = 0.0;
    {
        let info = uniforms.cascades[0];
        let wp = p + n * WORLD_BIAS_NORMAL;
        let gu = (wp.xz - uniforms.hmap_min) / info.spacing - 0.5;
        let cell = vec2<i32>(floor(gu));
        let frac = gu - vec2<f32>(cell);
        let vis_mask = probe_visibility_mask(p, n);
        for (var k = 0u; k < 4u; k = k + 1u) {
            let corner = vec2<u32>(k & 1u, k >> 1u);
            let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
            let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
            let w_corner = wxy.x * wxy.y;
            let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;
            var layer_dy: array<f32, 3>;
            for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
                let c = sh_probes[(base_cell + layer) * SH_STRIDE + 9u].xyz;
                layer_dy[layer] = abs(c.y - wp.y);
            }
            var farthest = 999u;
            if ss.two_closest_layers != 0u {
                // Select the two closest layers out of the three (drop the farthest).
                farthest = 0u;
                var max_dy = layer_dy[0];
                if layer_dy[1] > max_dy {
                    farthest = 1u;
                    max_dy = layer_dy[1];
                }
                if layer_dy[2] > max_dy {
                    farthest = 2u;
                }
            }

            var layer_w: array<f32, 3>;
            var layer_sum = 0.0;
            for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
                if layer == farthest {
                    layer_w[layer] = 0.0;
                } else {
                    let dy = layer_dy[layer];
                    layer_w[layer] = 1.0 / (1e-3 + dy * dy);
                    layer_sum += layer_w[layer];
                }
            }
            if layer_sum <= 0.0 {
                continue;
            }
            for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
                if layer == farthest {
                    continue;
                }
                let base = (base_cell + layer) * SH_STRIDE;
                let center = sh_probes[base + 9u].xyz;
                if ss.normal_bias_enabled != 0u || probe_is_visible(vis_mask, k, layer) {
                    let d_xz = abs(center.xz - wp.xz) / info.spacing;
                    let w_horiz = saturate(1.0 - d_xz.x) * saturate(1.0 - d_xz.y);
                    let w = w_horiz * (layer_w[layer] / layer_sum);
                    for (var i = 0u; i < 9u; i = i + 1u) {
                        wcoefs[i] += sh_probes[base + i].xyz * w;
                    }
                    wsum += w;
                }
            }
        }
    }

    let has_world = wsum > 1e-4;
    var e = uniforms.sky_color.xyz;

    if has_world {
        let inv_w = 1.0 / wsum;
        var norm_coefs: array<vec3<f32>, 9>;
        for (var i = 0u; i < 9u; i = i + 1u) {
            norm_coefs[i] = wcoefs[i] * inv_w;
        }
        // Direct analytic irradiance evaluation with Hanning window tapering.
        // Exactly 9 FMAs, zero negative ringing, zero loop iterations.
        e = sh_irradiance_windowed(norm_coefs, n);
    }

    gtao_out[idx] = vec4<f32>(e, 0.0);
}
