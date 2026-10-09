#import s_rc::heightmap_trace::{uniforms, probe_position_baked, probe_normal_baked, PROBES_PER_CELL}

@group(1) @binding(0) var<storage, read_write> probe_positions: array<vec4<f32>>;

fn probe_positions_for(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    if id >= info.size.x * info.size.y * PROBES_PER_CELL {
        return;
    }
    let layer = id % PROBES_PER_CELL;
    let cell_id = id / PROBES_PER_CELL;
    let probe = vec2<u32>(cell_id % info.size.x, cell_id / info.size.x);
    // Pairs: even words are (position, 1.0), odd words are the probe's normal
    // (see the `probe_positions` declaration in `heightmap_trace`).
    let i = info.probe_base + id;
    probe_positions[2 * i] = vec4<f32>(probe_position_baked(c, probe, layer), 1.0);
    probe_positions[2 * i + 1u] = vec4<f32>(probe_normal_baked(c, probe, layer), 0.0);
}

@compute @workgroup_size(64, 1, 1) fn probe_positions_0(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_1(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_2(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_3(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(3u, id.x); }
