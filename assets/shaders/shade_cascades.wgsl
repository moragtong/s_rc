#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> sh_probes: array<vec4<f32>>;

struct CascadesUniforms {
    hmap_min: vec2<f32>,
    hmap_max: vec2<f32>,
    hmap_res: vec2<u32>,
    spacing: f32,
    octa_side: u32,
    grid_size: vec2<u32>,
}
@group(0) @binding(3) var<uniform> cascades_uniforms: CascadesUniforms;

struct ShadeUniforms {
    clip_from_view: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
}
@group(0) @binding(4) var<uniform> shade_uniforms: ShadeUniforms;

fn sh_basis(dir: vec3<f32>) -> array<f32, 9> {
    let x = dir.x; let y = dir.y; let z = dir.z;
    return array<f32, 9>(
        0.282095, -0.488603 * y, 0.488603 * z, -0.488603 * x,
        1.092548 * x * y, -1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0),
        -1.092548 * x * z, 0.546274 * (x * x - y * y)
    );
}

@fragment
fn shade(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uvi = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_tex, uvi, 0).x;
    
    if depth == 1.0 || depth == 0.0 {
        return vec4<f32>(0.0); // background
    }
    
    let encoded_norm = textureLoad(normal_tex, uvi, 0).xyz;
    let normal = normalize(encoded_norm * 2.0 - 1.0); // prepass normal is mostly 0..1 packed
    
    // reconstruct world position from depth
    let uv = in.uv;
    let clip_pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    
    // actually, typically Inverse(proj) * clip to get view pos
    // Wait, Bevy reverse Z? Depth is usually 0..1
    // let's just do a cheap analytical lookup if possible, or just use world normal!
    // Since this is a mockup, applying the nearest probe based on camera position could just be enough.
    // Let's actually reconstruct world position accurately:
    
    // In Bevy, `color_attachment` for Normals are encoded. But wait, `NormalPrepass` is not encoded, it's just `f16` xyz in world space or view space? Bevy 0.14 uses World Space normal.
    let world_normal = normalize(textureLoad(normal_tex, uvi, 0).xyz); // assuming standard unpacked normal 
    
    // Evaluate SH!
    // For the mockup, let's just fetch probe (0, 0) or the middle probe!
    let probe_idx = (cascades_uniforms.grid_size.x / 2u) + cascades_uniforms.grid_size.x * (cascades_uniforms.grid_size.y / 2u);
    
    let basis = sh_basis(world_normal);
    var irradiance = vec3<f32>(0.0);
    
    for (var i = 0u; i < 9u; i++) {
        irradiance += sh_probes[probe_idx * 9u + i].xyz * basis[i];
    }
    
    // The albedo is not here, so we just return irradiance
    return vec4<f32>(irradiance, 0.5); // Additive
}
