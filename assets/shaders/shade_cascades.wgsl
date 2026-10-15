// Indirect diffuse from cascade 0's SH probes, added onto the lit HDR
// image. Evaluates the hybrid blend: bilinear horizontal interpolation in XZ
// and continuous inverse-square vertical weighting over the two closest probe layers.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import s_rc::heightmap_trace::{uniforms, sh_basis, oct_decode, trace, MISS, PI, SH_STRIDE, PROBES_PER_CELL, is_solid, texel_at, probe_visible, probe_visibility_mask, probe_is_visible, cell_direction}

@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var gbuffer: texture_2d<u32>;
// The blended probe set: `blend_sh` writes mix(previous relight, current
// relight, alpha) here every frame before this pass runs.
@group(1) @binding(2) var<storage, read> sh_probes: array<vec4<f32>>;

struct ShadeUniforms {
    view_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    indirect_enabled: u32,
    normal_bias_enabled: u32,
    use_vis_mask: u32,
    ao_enabled: u32,
}
@group(1) @binding(3) var<uniform> view_info: ShadeUniforms;
@group(1) @binding(4) var ao_tex: texture_2d<f32>;
@group(1) @binding(5) var radiance_min: texture_2d<f32>;
@group(1) @binding(6) var radiance_max: texture_2d<f32>;
@group(1) @binding(7) var<storage, read> rays: array<vec4<f32>>;

// Weight normalization epsilon - prevents division by near-zero when all probes are occluded.
const WEIGHT_EPS: f32 = 1e-4;
// Normal-biased fetch anchor offset.
const BIAS_NORMAL: f32 = 0.4;
const BIAS_VIEW: f32 = 0.0;

fn sh_coeff(base: u32, i: u32) -> vec3<f32> {
    return sh_probes[base + i].xyz;
}

// Analytic cosine-convolved L1 SH irradiance.
// Band 0: A_0 = PI, Band 1: A_1 = 2*PI/3 ~= 2.094395 (windowed to 1.5707963)
fn sh_irradiance(base: u32, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 2>(PI, 1.5707963);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let l = select(1u, 0u, i == 0u);
        e += sh_coeff(base, i) * (basis[i] * band[l]);
    }
    return max(e, vec3<f32>(0.0));
}

fn chebyshev_probe_visibility(p: vec3<f32>, c: vec3<f32>, probe_id: u32) -> f32 {
    let to_pixel = p - c;
    let dist_pixel = length(to_pixel);
    if dist_pixel < 1e-4 {
        return 1.0;
    }
    let dir = to_pixel / dist_pixel;

    let info = uniforms.cascades[0];
    let side = info.octa_side;
    let rays_per_probe = side * side;
    let ray_base = info.ray_base + probe_id * rays_per_probe;

    var sum_w = 0.0;
    var sum_d = 0.0;
    var sum_d2 = 0.0;

    for (var r = 0u; r < rays_per_probe; r++) {
        let ray_dir = cell_direction(r, side);
        let cos_theta = dot(dir, ray_dir);
        if cos_theta > 0.0 {
            let w = cos_theta * cos_theta;
            let ray = rays[ray_base + r];
            let hit_d = select(100.0, ray.a, ray.a >= 0.0);
            sum_w += w;
            sum_d += w * hit_d;
            sum_d2 += w * hit_d * hit_d;
        }
    }

    if sum_w < 1e-5 {
        return 1.0;
    }

    let mean = sum_d / sum_w;
    let mean_sq = sum_d2 / sum_w;
    let variance = max(mean_sq - mean * mean, 0.001);

    // 2cm depth bias to prevent surface acne
    let test_dist = max(0.0, dist_pixel - 0.02);
    if test_dist <= mean {
        return 1.0;
    }

    let diff = test_dist - mean;
    let p_unoccluded = variance / (variance + diff * diff);

    // Standard DDGI bleed reduction
    let bleed_bias = 0.2;
    let vis = clamp((p_unoccluded - bleed_bias) / (1.0 - bleed_bias), 0.0, 1.0);
    return vis * vis;
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

    if depth <= 0.0 || view_info.indirect_enabled == 0u {
        return vec4<f32>(0.0);
    }

    // Bevy's deferred G-buffer: r = sRGB base color + roughness, b = reflectance, metallic, occlusion, a = 24-bit octahedral normal + flags.
    let g = textureLoad(gbuffer, pixel, 0);
    let base_color = pow(unpack4x8unorm(g.r).rgb, vec3<f32>(2.2));
    let n = oct_decode(vec2<f32>(f32(g.a & 0xFFFu), f32((g.a >> 12u) & 0xFFFu)) / 4095.0);

    let info = uniforms.cascades[0];
    var fetch = p;
    if view_info.normal_bias_enabled != 0u {
        let view_dir = (ro - p) / max(max_t, 1e-4);
        fetch = p + n * BIAS_NORMAL + view_dir * BIAS_VIEW;
    }

    let gu = (fetch.xz - uniforms.hmap_min) / info.spacing - 0.5;
    let cell = vec2<i32>(floor(gu));
    let frac = gu - vec2<f32>(cell);

    var e_blend = vec3<f32>(0.0);
    var w_sum = 0.0;

    for (var k = 0u; k < 4u; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
        let wxy = mix(1.0 - frac, frac, vec2<f32>(corner));
        let w_corner = wxy.x * wxy.y;
        let base_cell = info.probe_base + (q.y * info.size.x + q.x) * PROBES_PER_CELL;

        var layer_dy: array<f32, 3>;
        for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
            let c = sh_probes[(base_cell + layer) * SH_STRIDE + 4u].xyz;
            layer_dy[layer] = abs(c.y - fetch.y);
        }

        // Two closest layers (drop the farthest of the three).
        var farthest = 0u;
        var max_dy = layer_dy[0];
        if layer_dy[1] > max_dy {
            farthest = 1u;
            max_dy = layer_dy[1];
        }
        if layer_dy[2] > max_dy {
            farthest = 2u;
        }

        var layer_w: array<f32, 3>;
        var layer_sum = 0.0;
        for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
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

        for (var layer = 0u; layer < PROBES_PER_CELL; layer++) {
            if layer == farthest {
                continue;
            }
            let probe_id = base_cell + layer;
            let base = probe_id * SH_STRIDE;
            let center = sh_probes[base + 4u].xyz;

            var vis: f32 = 1.0;
            if view_info.use_vis_mask != 0u {
                vis = chebyshev_probe_visibility(p, center, probe_id);
            } else {
                if !probe_visible(p, n, center) {
                    vis = 0.0;
                }
            }

            if vis > 0.0 {
                let w = w_corner * (layer_w[layer] / layer_sum) * vis;
                e_blend += w * sh_irradiance(base, n);
                w_sum += w;
            }
        }
    }

    var e = vec3<f32>(0.0);
    if w_sum > WEIGHT_EPS {
        e = e_blend / w_sum;
    }
    let ao = select(1.0, textureLoad(ao_tex, pixel, 0).r, view_info.ao_enabled != 0u);

    // Direct sun evaluation via dual heightmap visibility
    // Determine whether surface is closer to ceiling or floor layer
    let is_ceiling = n.y < -0.1 || (abs(n.y) <= 0.1 && p.y >= uniforms.split);
    let hmap_texel = clamp(texel_at(p.xz), vec2<i32>(0), vec2<i32>(uniforms.hmap_res) - 1);
    let sun_vis = select(
        textureLoad(radiance_max, hmap_texel, 0).a,
        textureLoad(radiance_min, hmap_texel, 0).a,
        is_ceiling,
    );
    let n_dot_l = max(dot(n, uniforms.sun_toward), 0.0);
    let direct_sun = (base_color / PI) * (uniforms.sun_irradiance * (n_dot_l * sun_vis));

    let indirect_diffuse = (base_color / PI * e) * ao;
    // Return alpha = 1.0 for solid geometry so it replaces the unshadowed direct light pass
    return vec4<f32>(direct_sun + indirect_diffuse, 1.0);
}
