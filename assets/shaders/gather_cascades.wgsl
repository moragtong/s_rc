@group(0) @binding(0) var heightmap: texture_2d<f32>;
@group(0) @binding(1) var metadata_min: texture_2d<vec4<f32>>;
@group(0) @binding(2) var metadata_max: texture_2d<vec4<f32>>;
@group(0) @binding(3) var<storage, read> hits: array<f32>;
@group(0) @binding(4) var<storage, read_write> sh_probes: array<vec4<f32>>;

struct Uniforms {
    hmap_min: vec2<f32>,
    hmap_max: vec2<f32>,
    hmap_res: vec2<u32>,
    spacing: f32,
    octa_side: u32,
    grid_size: vec2<u32>,
}
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    if n.x >= 0.0 { n.x -= t; } else { n.x += t; }
    if n.y >= 0.0 { n.y -= t; } else { n.y += t; }
    return normalize(n);
}

fn sh_basis(dir: vec3<f32>) -> array<f32, 9> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    return array<f32, 9>(
        0.282095,
        -0.488603 * y,
        0.488603 * z,
        -0.488603 * x,
        1.092548 * x * y,
        -1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        -1.092548 * x * z,
        0.546274 * (x * x - y * y)
    );
}

fn sample_radiance(hit_pos: vec3<f32>, ray_dy: f32) -> vec3<f32> {
    let hmap_size = uniforms.hmap_max - uniforms.hmap_min;
    let uv_x = (hit_pos.x - uniforms.hmap_min.x) / hmap_size.x;
    let uv_y = (hit_pos.z - uniforms.hmap_min.y) / hmap_size.y;
    
    if uv_x < 0.0 || uv_x > 1.0 || uv_y < 0.0 || uv_y > 1.0 {
        return vec3<f32>(0.0); // Out of bounds
    }
    
    let tex_x = u32(clamp(uv_x * f32(uniforms.hmap_res.x), 0.0, f32(uniforms.hmap_res.x - 1u)));
    let tex_y = u32(clamp(uv_y * f32(uniforms.hmap_res.y), 0.0, f32(uniforms.hmap_res.y - 1u)));
    
    var metadata = vec4<f32>(0.0);
    if ray_dy > 0.0 {
        metadata = textureLoad(metadata_min, vec2<u32>(tex_x, tex_y), 0);
    } else {
        metadata = textureLoad(metadata_max, vec2<u32>(tex_x, tex_y), 0);
    }
    
    let normal = normalize(metadata.xyz);
    let albedo = metadata.w; // simplified to scalar luminance/base shade
    
    // Simple mock lighting (sun light from (-1, 0.75, -0.35))
    let sun_dir = normalize(vec3<f32>(-1.0, 0.75, -0.35));
    let NdotL = max(0.0, dot(normal, sun_dir));
    let radiance = vec3<f32>(albedo * NdotL * 3.0); // Mock sun intensity
    
    return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn gather(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe_x = global_id.x;
    let probe_z = global_id.y;
    
    if probe_x >= uniforms.grid_size.x || probe_z >= uniforms.grid_size.y {
        return;
    }
    
    let w_pos_x = (f32(probe_x) + 0.5) * uniforms.spacing;
    let w_pos_z = (f32(probe_z) + 0.5) * uniforms.spacing;
    
    let hmap_size = uniforms.hmap_max - uniforms.hmap_min;
    let uv_x = (w_pos_x - uniforms.hmap_min.x) / hmap_size.x;
    let uv_y = (w_pos_z - uniforms.hmap_min.y) / hmap_size.y;
    
    let tex_x = u32(clamp(uv_x * f32(uniforms.hmap_res.x), 0.0, f32(uniforms.hmap_res.x - 1u)));
    let tex_y = u32(clamp(uv_y * f32(uniforms.hmap_res.y), 0.0, f32(uniforms.hmap_res.y - 1u)));
    let p_min_max = textureLoad(heightmap, vec2<u32>(tex_x, tex_y), 0);
    
    let probe_y = p_min_max.y + 0.05; // Base probe origin setup
    let origin = vec3<f32>(w_pos_x, probe_y, w_pos_z);
    
    let rays_per_probe = uniforms.octa_side * uniforms.octa_side;
    let probe_idx = probe_z * uniforms.grid_size.x + probe_x;
    let probe_hit_base = probe_idx * rays_per_probe;
    
    // Clear SH probe buffers initially before accumulating
    for (var i = 0u; i < 9u; i++) {
        sh_probes[probe_idx * 9u + i] = vec4<f32>(0.0);
    }
    
    // Wait for all threads to clear if we were shared, but they are localized.
    // Integrate incoming radiance weighted by Spherical Harmonics bases
    
    for (var r = 0u; r < rays_per_probe; r++) {
        let rx = r % uniforms.octa_side;
        let ry = r / uniforms.octa_side;
        let ruv = vec2<f32>(f32(rx) + 0.5, f32(ry) + 0.5) / f32(uniforms.octa_side);
        let dir = oct_decode(ruv);
        
        let hit_d = hits[probe_hit_base + r];
        
        var radiance = vec3<f32>(0.0);
        if hit_d < 1000.0 {
            let hit_pos = origin + dir * hit_d;
            radiance = sample_radiance(hit_pos, dir.y);
        }
        
        let basis = sh_basis(dir);
        let weight = (4.0 * 3.14159) / f32(rays_per_probe);
        
        for (var i = 0u; i < 9u; i++) {
            let contrib = radiance * basis[i] * weight;
            // Since there's one thread per probe sequentially looping over rays, no atomics strictly needed
            let current = sh_probes[probe_idx * 9u + i].xyz;
            sh_probes[probe_idx * 9u + i] = vec4<f32>(current + contrib, 1.0);
        }
    }
}
