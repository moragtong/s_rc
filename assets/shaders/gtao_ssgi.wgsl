// Smooth Hemisphere-Integrated Probe GI with hybrid horizontal-bilinear and
// inverse-square vertical blend, composited with contact SSAO.
//
// One thread per pixel. For the pixel's surface point `p` and its normal `n`:
// 1. Gathers the 12 cascade-0 world probes (4 corner cells x 3 vertical layers),
//    gated by `probe_visible(p, n, center)`.
// 2. Evaluates the hybrid blend: bilinear horizontal interpolation in XZ and
//    continuous inverse-square vertical weighting (1 / (eps + dy^2)).
// 3. Numerically integrates clamped radiance (max(sh_radiance, 0)) across 6
//    azimuth slices around the normal hemisphere at 45 degree elevation.
//    Clamping before integration avoids negative ringing lobes inherent to SH.
// 4. Writes irradiance to `gtao_out`, which is multiplied by Bevy's XeGTAO SSAO
//    in `composite_indirect` for clean contact shadowing.

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
    normal_bias_enabled: u32,
}

const TAU: f32 = 6.28318530718;
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

// Radiance from direction `d` reconstructed from SH radiance coefficients.
// Clamped to non-negative to eliminate SH ringing / negative lobes.
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
                if ss.normal_bias_enabled != 0u || probe_visible(p, n, c) {
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

    // Renormalize coefficients over the surviving visible probes
    var norm_coefs: array<vec3<f32>, 9>;
    if has_world {
        let inv_w = 1.0 / wsum;
        for (var i = 0u; i < 9u; i = i + 1u) {
            norm_coefs[i] = wcoefs[i] * inv_w;
        }
    }

    // Directional numerical integration over the normal hemisphere.
    // 6 azimuthal slices at elevation angle 45 degrees (sin^2 theta = 0.5).
    // Because the integrand is an order-2 spherical harmonic, a 6-tap uniform
    // quadrature integrates azimuth frequencies up to 5 with exact accuracy.
    let slices = max(ss.slices, 1u);
    let dphi = TAU / f32(slices);
    let rem = 0.5 * dphi;
    let s2_mid = 0.5;
    let c_mid = sqrt(0.5);

    var e = vec3<f32>(0.0);
    for (var m = 0u; m < slices; m = m + 1u) {
        let phi = (f32(m) + 0.5) * dphi;
        let tdir = normalize(tan_x * cos(phi) + tan_y * sin(phi));
        let band_dir = normalize(n * sqrt(s2_mid) + tdir * c_mid);
        var radiance = sky;
        if has_world {
            radiance = sh_radiance(norm_coefs, band_dir);
        }
        e += radiance * rem;
    }

    gtao_out[idx] = vec4<f32>(e, 0.0);
}
