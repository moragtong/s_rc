// Sunlit outgoing radiance of every heightmap texel's floor and ceiling,
// shadowed by tracing toward the sun through the heightmap. Cascade rays
// read it at their hits. Supports iterative multi-bounce GI by evaluating
// the latest probe irradiance and adding it to the surface radiance.

#import s_rc::heightmap_trace::{
    heightmap, uniforms, trace, cell_size, MISS, PI, SH_STRIDE, PROBES_PER_CELL,
    probe_visibility_mask, probe_is_visible, sh_basis, cell_direction,
}

@group(1) @binding(0) var normal_min: texture_2d<f32>;
@group(1) @binding(1) var normal_max: texture_2d<f32>;
@group(1) @binding(2) var albedo_min: texture_2d<f32>;
@group(1) @binding(3) var albedo_max: texture_2d<f32>;
@group(1) @binding(4) var radiance_min: texture_storage_2d<rgba32float, write>;
@group(1) @binding(5) var radiance_max: texture_storage_2d<rgba32float, write>;
@group(1) @binding(6) var<storage, read> sh_probes: array<vec4<f32>>;

struct LightUniforms {
    bounce_index: u32,
    temporal_bounce: u32,
    temporal_weight: f32,
    pad0: u32,
}
@group(1) @binding(7) var<uniform> light_info: LightUniforms;
@group(1) @binding(8) var<storage, read> bounce_history_in: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read_write> bounce_history_out: array<vec4<f32>>;
@group(1) @binding(10) var<storage, read> rays: array<vec4<f32>>;

// Shadow rays start this far off the surface, m.
// Matches SHADOW_RAY_BIAS in heightmap_trace.wgsl.
const SHADOW_BIAS: f32 = 0.02;
const WEIGHT_EPS: f32 = 1e-4;

struct SunlitResult {
    radiance: vec3<f32>,
    visible: f32,
}

fn sunlit(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> SunlitResult {
    let cos_l = dot(n, uniforms.sun_toward);
    if cos_l <= 0.0 {
        return SunlitResult(vec3<f32>(0.0), 0.0);
    }
    if trace(p + n * SHADOW_BIAS, uniforms.sun_toward, 0.0, 1.0e4).t != MISS {
        return SunlitResult(vec3<f32>(0.0), 0.0);
    }
    return SunlitResult(albedo / PI * uniforms.sun_irradiance * cos_l, 1.0);
}

fn sh_coeff(base: u32, i: u32) -> vec3<f32> {
    return sh_probes[base + i].xyz;
}

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

fn evaluate_probe_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let info = uniforms.cascades[0];
    let fetch = p + n * 0.2;
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
            let vis = chebyshev_probe_visibility(p, center, probe_id);
            if vis > 0.0 {
                let w = w_corner * (layer_w[layer] / layer_sum) * vis;
                e_blend += w * sh_irradiance(base, n);
                w_sum += w;
            }
        }
    }

    if w_sum > WEIGHT_EPS {
        return e_blend / w_sum;
    }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn surface_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= uniforms.hmap_res) {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let xz = (vec2<f32>(id.xy) + 0.5) * cell_size() + uniforms.hmap_min;
    let h = textureLoad(heightmap, texel, 0).xy;

    let texel_idx = id.y * uniforms.hmap_res.x + id.x;

    var ceiling = vec3<f32>(0.0);
    var vis_c = 0.0;
    let n_ceiling = textureLoad(normal_min, texel, 0).xyz;
    if dot(n_ceiling, n_ceiling) > 0.0 {
        let norm_c = normalize(n_ceiling);
        let albedo_c = textureLoad(albedo_min, texel, 0).rgb;
        let pos_c = vec3<f32>(xz.x, h.x, xz.y);
        let res_c = sunlit(pos_c, norm_c, albedo_c);
        ceiling = res_c.radiance;
        vis_c = res_c.visible;
        if light_info.bounce_index > 0u {
            ceiling += (albedo_c / PI) * evaluate_probe_irradiance(pos_c, norm_c);
        } else if light_info.temporal_bounce > 0u && light_info.temporal_weight > 0.0 {
            let indirect = (albedo_c / PI) * evaluate_probe_irradiance(pos_c, norm_c);
            let accum = light_info.temporal_weight * indirect;
            bounce_history_out[texel_idx * 2u + 0u] = vec4<f32>(accum, 0.0);
            ceiling += accum;
        } else {
            bounce_history_out[texel_idx * 2u + 0u] = vec4<f32>(0.0);
        }
    } else if light_info.bounce_index == 0u {
        bounce_history_out[texel_idx * 2u + 0u] = vec4<f32>(0.0);
    }

    var floor_l = vec3<f32>(0.0);
    var vis_f = 0.0;
    let n_floor = textureLoad(normal_max, texel, 0).xyz;
    if dot(n_floor, n_floor) > 0.0 {
        let norm_f = normalize(n_floor);
        let albedo_f = textureLoad(albedo_max, texel, 0).rgb;
        let pos_f = vec3<f32>(xz.x, h.y, xz.y);
        let res_f = sunlit(pos_f, norm_f, albedo_f);
        floor_l = res_f.radiance;
        vis_f = res_f.visible;
        if light_info.bounce_index > 0u {
            floor_l += (albedo_f / PI) * evaluate_probe_irradiance(pos_f, norm_f);
        } else if light_info.temporal_bounce > 0u && light_info.temporal_weight > 0.0 {
            let indirect = (albedo_f / PI) * evaluate_probe_irradiance(pos_f, norm_f);
            let accum = light_info.temporal_weight * indirect;
            bounce_history_out[texel_idx * 2u + 1u] = vec4<f32>(accum, 0.0);
            floor_l += accum;
        } else {
            bounce_history_out[texel_idx * 2u + 1u] = vec4<f32>(0.0);
        }
    } else if light_info.bounce_index == 0u {
        bounce_history_out[texel_idx * 2u + 1u] = vec4<f32>(0.0);
    }

    // Alpha channel stores direct sunlight visibility (1.0 = visible to sun, 0.0 = shadowed)
    textureStore(radiance_min, texel, vec4<f32>(ceiling, vis_c));
    textureStore(radiance_max, texel, vec4<f32>(floor_l, vis_f));
}
