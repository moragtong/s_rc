#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_bindings::mesh
#import bevy_pbr::mesh_vertex_output::MeshVertexOutput

@group(3) @binding(0) var heightmap_tex: texture_2d<f32>;
@group(3) @binding(1) var heightmap_samp: sampler;

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
    let max_h = ht.y;
    
    var local_pos = vertex.position;
    // displace up by max_h
    local_pos.y += max_h;
    
    out.uv = uv;
    out.position = mesh_position_local_to_clip(get_world_from_local(vertex.instance_index), vec4<f32>(local_pos, 1.0));
    return out;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = mesh.uv;
    let ht = textureSample(heightmap_tex, heightmap_samp, uv);
    let r = max(0.0, ht.x / 4.0);
    let g = max(0.0, ht.y / 4.0);
    return vec4<f32>(r, g, 0.0, 0.5); // semi transparent
}
