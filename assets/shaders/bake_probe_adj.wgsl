#import s_rc::heightmap_trace::{
    uniforms, heightmap, winding, is_solid, texel_at, probe_position, PROBES_PER_CELL
}

@group(1) @binding(0) var<storage, read_write> probe_adj_out: array<u32>;

const OFFSETS = array<vec2<i32>, 9>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0), vec2<i32>(0,  0), vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1),
);

fn probe_to_probe_visible(p0: vec3<f32>, p1: vec3<f32>) -> bool {
    let to_probe = p1 - p0;
    let dist_xz = length(to_probe.xz);
    if dist_xz <= 1e-4 {
        return true;
    }
    let steps = 6u;
    for (var i = 0u; i <= steps; i = i + 1u) {
        let t = f32(i) / f32(steps);
        let sample_pos = p0 + to_probe * t;
        let sample_texel = texel_at(sample_pos.xz);
        if is_solid(sample_texel) {
            return false;
        }
        let h = textureLoad(heightmap, vec2<u32>(sample_texel), 0).xy;
        if h.y > uniforms.split - 9.0 && sample_pos.y < h.y - 0.25 {
            return false;
        }
        if h.x < 5000.0 && sample_pos.y > h.x + 0.25 {
            return false;
        }
    }
    return true;
}

fn bake_adj_for(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    if id >= info.size.x * info.size.y * PROBES_PER_CELL {
        return;
    }
    let layer = id % PROBES_PER_CELL;
    let cell_id = id / PROBES_PER_CELL;
    let px = cell_id % info.size.x;
    let py = cell_id / info.size.x;

    let probe_idx = info.probe_base + id;
    let center_pos = probe_position(c, vec2<u32>(px, py), layer);

    var mask = 0u;

    for (var n = 0u; n < 9u; n = n + 1u) {
        if n == 4u {
            // Center probe is always visible to itself
            mask = mask | (1u << n);
            continue;
        }

        let qx = clamp(i32(px) + OFFSETS[n].x, 0, i32(info.size.x) - 1);
        let qy = clamp(i32(py) + OFFSETS[n].y, 0, i32(info.size.y) - 1);

        if qx == i32(px) && qy == i32(py) {
            // Clamped to grid border (replicated edge)
            mask = mask | (1u << n);
            continue;
        }

        let q_pos = probe_position(c, vec2<u32>(u32(qx), u32(qy)), layer);

        if probe_to_probe_visible(center_pos, q_pos) {
            mask = mask | (1u << n);
        }
    }

    probe_adj_out[probe_idx] = mask;
}

@compute @workgroup_size(64, 1, 1) fn bake_adj_0(@builtin(global_invocation_id) id: vec3<u32>) { bake_adj_for(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn bake_adj_1(@builtin(global_invocation_id) id: vec3<u32>) { bake_adj_for(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn bake_adj_2(@builtin(global_invocation_id) id: vec3<u32>) { bake_adj_for(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn bake_adj_3(@builtin(global_invocation_id) id: vec3<u32>) { bake_adj_for(3u, id.x); }
