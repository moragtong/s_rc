// Indirect diffuse from cascade 0's SH probes, added onto the lit HDR
// image. Each of the 4 bilinear probes is parallax corrected: its
// surroundings are taken to lie on a sphere of its radius (its farthest
// hit) around it, so the SH is read in the direction from the probe to
// where the shading point's ray would meet that sphere.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import s_rc::heightmap_trace::{uniforms, sh_basis, oct_decode, nonzero, PI, SH_STRIDE}

@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var gbuffer: texture_2d<u32>;
@group(1) @binding(2) var<storage, read> sh_probes: array<vec4<f32>>;

struct ShadeUniforms {
    view_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    probes_enabled: u32,
    indirect_enabled: u32,
}
@group(1) @binding(3) var<uniform> view_info: ShadeUniforms;

// Cosine-distributed directions per probe for the irradiance integral.
const HEMISPHERE_SAMPLES: u32 = 32u;
const GOLDEN_ANGLE: f32 = 2.39996323;

fn sh_radiance(base: u32, d: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(d);
    var l = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        l += sh_probes[base + i].xyz * basis[i];
    }
    return max(l, vec3<f32>(0.0));
}

// Irradiance from the SH convolved with the clamped cosine (no parallax).
fn sh_irradiance(base: u32, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 3>(PI, 2.094395, 0.785398);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let l = select(select(2u, 1u, i < 4u), 0u, i == 0u);
        e += sh_probes[base + i].xyz * (basis[i] * band[l]);
    }
    return max(e, vec3<f32>(0.0));
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

fn probe_irradiance(probe: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let base = probe * SH_STRIDE;
    let center = sh_probes[base + 9u];
    let oc = p - center.xyz;
    let radius = center.w;
    let frame = tangent_frame(n);
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < HEMISPHERE_SAMPLES; k++) {
        let u = (f32(k) + 0.5) / f32(HEMISPHERE_SAMPLES);
        let r = sqrt(u);
        let phi = f32(k) * GOLDEN_ANGLE;
        let w = frame * vec3<f32>(r * cos(phi), r * sin(phi), sqrt(1.0 - u));
        var d = w;
        // Far root of |p + t w - center| = radius. Probes that hit nothing
        // have radius 0 and are read uncorrected.
        if radius > 0.0 {
            let b = dot(oc, w);
            let disc = b * b - (dot(oc, oc) - radius * radius);
            if disc >= 0.0 {
                let t = -b + sqrt(disc);
                if t > 0.0 {
                    d = normalize(oc + w * t);
                }
            }
        }
        e += sh_radiance(base, d);
    }
    return e * (PI / f32(HEMISPHERE_SAMPLES));
}

// Probe spheres of every cascade along the view ray, shaded as white
// diffuse balls from their SH; returns (color, distance) or distance < 0.
fn probe_spheres(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> vec4<f32> {
    var best = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    for (var c = 0u; c < uniforms.cascade_count; c++) {
        let info = uniforms.cascades[c];
        let s = info.spacing;
        let radius = 0.15 * s;
        // 2D DDA over the probe grid, testing each column's probe.
        let g = (ro.xz - uniforms.hmap_min) / s;
        var cell = vec2<i32>(floor(g));
        let dir_step = select(vec2<i32>(-1), vec2<i32>(1), rd.xz >= vec2<f32>(0.0));
        let inv = 1.0 / nonzero(rd.xz);
        let edge = (vec2<f32>(cell) + select(vec2<f32>(0.0), vec2<f32>(1.0), dir_step > vec2<i32>(0))) * s
            + uniforms.hmap_min;
        var t_max = (edge - ro.xz) * inv;
        let t_delta = abs(s * inv);
        for (var i = 0u; i < 256u; i++) {
            if all(cell >= vec2<i32>(0)) && all(cell < vec2<i32>(info.size)) {
                let probe = info.probe_base + u32(cell.y) * info.size.x + u32(cell.x);
                let base = probe * SH_STRIDE;
                let center = sh_probes[base + 9u].xyz;
                let oc = ro - center;
                let b = dot(oc, rd);
                let disc = b * b - (dot(oc, oc) - radius * radius);
                if disc >= 0.0 {
                    let t = -b - sqrt(disc);
                    if t > 0.0 && t < max_t && (best.w < 0.0 || t < best.w) {
                        let n = normalize(ro + rd * t - center);
                        best = vec4<f32>(sh_irradiance(base, n) / PI, t);
                    }
                }
            }
            if min(t_max.x, t_max.y) > max_t {
                break;
            }
            if t_max.x < t_max.y {
                t_max.x += t_delta.x;
                cell.x += dir_step.x;
            } else {
                t_max.y += t_delta.y;
                cell.y += dir_step.y;
            }
        }
    }
    return best;
}

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

    if view_info.probes_enabled != 0u {
        var rd: vec3<f32>;
        if depth > 0.0 {
            rd = (p - ro) / max_t;
        } else {
            let clip = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
            let view = view_info.view_from_clip * clip;
            rd = normalize((view_info.world_from_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz - ro);
        }
        let sphere = probe_spheres(ro, rd, max_t);
        if sphere.w >= 0.0 {
            return vec4<f32>(sphere.rgb, 0.0);
        }
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
    let u = (p.xz - uniforms.hmap_min) / info.spacing - 0.5;
    let cell = vec2<i32>(floor(u));
    let f = u - floor(u);
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let wxy = select(1.0 - f, f, corner == vec2<u32>(1u));
        let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
        let probe = info.probe_base + q.y * info.size.x + q.x;
        e += (wxy.x * wxy.y) * probe_irradiance(probe, p, n);
    }
    return vec4<f32>(base_color * (1.0 - metallic) / PI * e, 0.0);
}
