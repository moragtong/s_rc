// Spherical Harmonics Parallax Shading (Irradiance Warp at Render Points)

@group(0) @binding(0) var g_buffer_depth: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> hits: array<f32>;
// We'll add the color/lighting volumes next

@compute @workgroup_size(8, 8, 1)
fn shade(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1 thread = 1 pixel
    let pixel_coords = vec2<u32>(global_id.x, global_id.y);
    
    // Read depth -> reconstruct world pos
    // Fetch SH probes around world pos
    
    // NOTE: This is where we implement Parallax-Corrected Spherical Harmonics
    // using the proxy bounds fetched from hits.
}
