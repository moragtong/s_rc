// Gather phase: Compute radiance at ray hits and accumulate into SH

@group(0) @binding(0) var hit_distances: texture_2d<f32>;
@group(0) @binding(1) var metadata_max: texture_2d<vec4<f32>>;
@group(0) @binding(2) var metadata_min: texture_2d<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sh_probes: array<vec4<f32>>; // SH coefficients

@compute @workgroup_size(8, 8, 1)
fn gather(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Scaffold for calculating World Position from hits
    // Sampling metadata min/max depending on ray direction
    // Transforming Radiance into Spherical Harmonics
}
