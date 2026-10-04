#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_bindings::mesh
#import bevy_pbr::mesh_vertex_output::MeshVertexOutput

@group(3) @binding(0) var heightmap_tex: texture_2d<f32>;
@group(3) @binding(1) var heightmap_samp: sampler;
@group(3) @binding(2) var<uniform> channel: f32;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var uv = vertex.uv;
    
    // fetch height
    let ht = textureSampleLevel(heightmap_tex, heightmap_samp, uv, 0.0);
    
    var local_pos = vertex.position;
    if channel > 0.5 {
        local_pos.y += ht.y;
    } else {
        // Open-sky columns have their ceiling far up; draw it at 15 m.
        local_pos.y += min(ht.x, 15.0);
    }
    
    out.uv = uv;
    out.position = mesh_position_local_to_clip(get_world_from_local(vertex.instance_index), vec4<f32>(local_pos, 1.0));
    return out;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = mesh.uv;
    let ht = textureSample(heightmap_tex, heightmap_samp, uv);
    if channel > 0.5 {
        let g = max(0.0, ht.y / 4.0);
        return vec4<f32>(0.0, g + 0.2, 0.0, 0.5); // green for ground
    } else {
        let r = max(0.0, ht.x / 4.0);
        return vec4<f32>(r + 0.2, 0.0, 0.0, 0.5); // red for roof
    }
}
