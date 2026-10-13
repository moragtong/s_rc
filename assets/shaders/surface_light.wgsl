// Sunlit outgoing radiance of every heightmap texel's floor and ceiling,
// shadowed by marching toward the sun using the Aaltonen / Quilez (2018)
// soft shadow estimator against the dual heightmap and winding map.
// Cascade rays read it at their hits.

#import s_rc::heightmap_trace::{heightmap, uniforms, cell_size, PI, sun_visibility}

@group(1) @binding(0) var normal_min: texture_2d<f32>;
@group(1) @binding(1) var normal_max: texture_2d<f32>;
@group(1) @binding(2) var albedo_min: texture_2d<f32>;
@group(1) @binding(3) var albedo_max: texture_2d<f32>;
@group(1) @binding(4) var radiance_min: texture_storage_2d<rgba32float, write>;
@group(1) @binding(5) var radiance_max: texture_storage_2d<rgba32float, write>;

// Shadow rays start this far off the surface, m.
// Matches SHADOW_RAY_BIAS in heightmap_trace.wgsl.
const SHADOW_BIAS: f32 = 0.02;

fn sunlit(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let cos_l = dot(n, uniforms.sun_toward);
    if cos_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let vis = sun_visibility(p + n * SHADOW_BIAS, uniforms.sun_toward);
    return albedo / PI * uniforms.sun_irradiance * cos_l * vis;
}

@compute @workgroup_size(8, 8, 1)
fn surface_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= uniforms.hmap_res) {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let xz = (vec2<f32>(id.xy) + 0.5) * cell_size() + uniforms.hmap_min;
    let h = textureLoad(heightmap, texel, 0).xy;

    // Texels the bake found no surface in have a zero normal.
    var ceiling = vec3<f32>(0.0);
    let n_ceiling = textureLoad(normal_min, texel, 0).xyz;
    if dot(n_ceiling, n_ceiling) > 0.0 {
        let albedo = textureLoad(albedo_min, texel, 0).rgb;
        ceiling = sunlit(vec3<f32>(xz.x, h.x, xz.y), normalize(n_ceiling), albedo);
    }
    var floor_l = vec3<f32>(0.0);
    let n_floor = textureLoad(normal_max, texel, 0).xyz;
    if dot(n_floor, n_floor) > 0.0 {
        let albedo = textureLoad(albedo_max, texel, 0).rgb;
        floor_l = sunlit(vec3<f32>(xz.x, h.y, xz.y), normalize(n_floor), albedo);
    }
    textureStore(radiance_min, texel, vec4<f32>(ceiling, 1.0));
    textureStore(radiance_max, texel, vec4<f32>(floor_l, 1.0));
}
