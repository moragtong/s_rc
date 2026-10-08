#import s_rc::heightmap_trace::{uniforms, probe_position_baked}

@group(1) @binding(0) var<storage, read_write> probe_positions: array<vec4<f32>>;

fn probe_positions_for(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    if id >= info.size.x * info.size.y {
        return;
    }
    let probe = vec2<u32>(id % info.size.x, id / info.size.x);
    probe_positions[info.probe_base + id] = vec4<f32>(probe_position_baked(c, probe), 1.0);
}

@compute @workgroup_size(64, 1, 1) fn probe_positions_0(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_1(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_2(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn probe_positions_3(@builtin(global_invocation_id) id: vec3<u32>) { probe_positions_for(3u, id.x); }
