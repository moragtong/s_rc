#import s_rc::heightmap_trace::{
    uniforms, heightmap, ceiling_normal, floor_normal, winding,
    is_solid, texel_at, probe_position, probe_visible, PROBES_PER_CELL
}

@group(1) @binding(0) var probe_vis_out: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.xy;
    if t.x >= uniforms.hmap_res.x || t.y >= uniforms.hmap_res.y {
        return;
    }

    let half_spacing = uniforms.cascades[0].spacing * 0.5;
    let uv = (vec2<f32>(t) + 0.5) / vec2<f32>(uniforms.hmap_res);
    let xz = uniforms.hmap_min + uv * (uniforms.hmap_max - uniforms.hmap_min) + vec2<f32>(half_spacing, half_spacing);

    let sample_t = texel_at(xz);
    let solid = is_solid(sample_t);
    if solid {
        textureStore(probe_vis_out, vec2<u32>(t), vec4<u32>(0u));
        return;
    }

    let h = textureLoad(heightmap, vec2<u32>(sample_t), 0).xy;
    let has_floor = h.y > uniforms.split - 9.0;
    let has_ceil = h.x < 5000.0;

    let y_floor = select(uniforms.split - 0.5, h.y, has_floor);
    let y_ceil = select(uniforms.split + 0.5, h.x, has_ceil);

    var n_floor = textureLoad(floor_normal, vec2<u32>(sample_t), 0).xyz;
    if dot(n_floor, n_floor) < 0.1 {
        n_floor = vec3<f32>(0.0, 1.0, 0.0);
    }
    let p_floor = vec3<f32>(xz.x, y_floor, xz.y);

    var n_ceil = textureLoad(ceiling_normal, vec2<u32>(sample_t), 0).xyz;
    if dot(n_ceil, n_ceil) < 0.1 {
        n_ceil = vec3<f32>(0.0, -1.0, 0.0);
    }
    let p_ceil = vec3<f32>(xz.x, y_ceil, xz.y);

    let info = uniforms.cascades[0];
    let gu = (xz - uniforms.hmap_min) / info.spacing - 0.5;
    let cell = vec2<i32>(floor(gu));

    var floor_mask = 0u;
    var ceil_mask = 0u;

    for (var k = 0u; k < 4u; k = k + 1u) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let q = vec2<u32>(clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(info.size) - 1));
        for (var layer = 0u; layer < PROBES_PER_CELL; layer = layer + 1u) {
            let center = probe_position(0u, q, layer);
            let bit_idx = k * 3u + layer;

            if has_floor {
                if probe_visible(p_floor, n_floor, center) {
                    floor_mask = floor_mask | (1u << bit_idx);
                }
            } else {
                if probe_visible(p_floor, vec3<f32>(0.0, 1.0, 0.0), center) {
                    floor_mask = floor_mask | (1u << bit_idx);
                }
            }

            if has_ceil {
                if probe_visible(p_ceil, n_ceil, center) {
                    ceil_mask = ceil_mask | (1u << bit_idx);
                }
            } else {
                if probe_visible(p_ceil, vec3<f32>(0.0, -1.0, 0.0), center) {
                    ceil_mask = ceil_mask | (1u << bit_idx);
                }
            }
        }
    }

    let packed = floor_mask | (ceil_mask << 16u);
    textureStore(probe_vis_out, vec2<u32>(t), vec4<u32>(packed, 0u, 0u, 0u));
}
