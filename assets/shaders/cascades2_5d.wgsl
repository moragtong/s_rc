@group(0) @binding(0) var heightmap: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> hits: array<f32>;
// Using f32 for hits to record raw distances

struct Uniforms {
    hmap_min: vec2<f32>,
    hmap_max: vec2<f32>,
    hmap_res: vec2<u32>,
    spacing: f32,
    octa_side: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    if n.x >= 0.0 { n.x -= t; } else { n.x += t; }
    if n.y >= 0.0 { n.y -= t; } else { n.y += t; }
    return normalize(n);
}

@compute @workgroup_size(8, 8, 1)
fn cascade0(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe_x = global_id.x;
    let probe_z = global_id.y;
    
    // We are running 1 thread per probe! 
    // And inside we loop over rays. Or better: 2D compute where X=probe, Y=ray?
    // Let's do 1 thread per PROBE for now, and loop over 64 rays.
    
    let w_pos_x = (f32(probe_x) + 0.5) * uniforms.spacing; // relative to some origin
    let w_pos_z = (f32(probe_z) + 0.5) * uniforms.spacing;
    
    // Get Hmap UV 
    let hmap_size = uniforms.hmap_max - uniforms.hmap_min;
    
    let uv_x = (w_pos_x - uniforms.hmap_min.x) / hmap_size.x;
    let uv_y = (w_pos_z - uniforms.hmap_min.y) / hmap_size.y;
    
    // Fetch hmap max_h for the probe's Y
    // For exact texel fetch:
    let tex_x = u32(clamp(uv_x * f32(uniforms.hmap_res.x), 0.0, f32(uniforms.hmap_res.x - 1u)));
    let tex_y = u32(clamp(uv_y * f32(uniforms.hmap_res.y), 0.0, f32(uniforms.hmap_res.y - 1u)));
    let p_min_max = textureLoad(heightmap, vec2<u32>(tex_x, tex_y), 0);
    
    let probe_y = p_min_max.y + 0.05; // slightly above the max_h
    let origin = vec3<f32>(w_pos_x, probe_y, w_pos_z);
    
    let rays_per_probe = uniforms.octa_side * uniforms.octa_side;
    let probe_hit_base = (probe_z * 64u + probe_x) * rays_per_probe; // dummy
    
    for (var r = 0u; r < rays_per_probe; r++) {
        let rx = r % uniforms.octa_side;
        let ry = r / uniforms.octa_side;
        let ruv = vec2<f32>(f32(rx) + 0.5, f32(ry) + 0.5) / f32(uniforms.octa_side);
        let dir = oct_decode(ruv);
        
        var hit_d = 1000.0; // miss
        
        // 2D DDA in XZ
        let ray_dx = dir.x;
        let ray_dz = dir.z;
        let ray_dy = dir.y;
        
        let cell_size_x = hmap_size.x / f32(uniforms.hmap_res.x);
        let cell_size_z = hmap_size.y / f32(uniforms.hmap_res.y);
        
        var cur_tx = tex_x;
        var cur_tz = tex_y;
        
        let step_x = select(-1, 1, ray_dx > 0.0);
        let step_z = select(-1, 1, ray_dz > 0.0);
        
        let t_delta_x = abs(cell_size_x / ray_dx);
        let t_delta_z = abs(cell_size_z / ray_dz);
        
        let min_b_x = uniforms.hmap_min.x + f32(cur_tx) * cell_size_x;
        let max_b_x = uniforms.hmap_min.x + f32(cur_tx + 1u) * cell_size_x;
        let min_b_z = uniforms.hmap_min.y + f32(cur_tz) * cell_size_z;
        let max_b_z = uniforms.hmap_min.y + f32(cur_tz + 1u) * cell_size_z;
        
        var t_max_x = select((origin.x - min_b_x) / -ray_dx, (max_b_x - origin.x) / ray_dx, ray_dx > 0.0);
        var t_max_z = select((origin.z - min_b_z) / -ray_dz, (max_b_z - origin.z) / ray_dz, ray_dz > 0.0);
        
        var current_t = 0.0;
        
        for (var step = 0; step < 128; step++) {
            
            // fetch heightmap at cur_tx, cur_tz
            let min_max = textureLoad(heightmap, vec2<u32>(cur_tx, cur_tz), 0);
            let h_min = min_max.x;
            let h_max = min_max.y;
            
            // next intersection in XZ plane
            let next_t = min(t_max_x, t_max_z);
            
            // Ray interval in Y
            let y_in = origin.y + current_t * ray_dy;
            let y_out = origin.y + next_t * ray_dy;
            
            let min_y = min(y_in, y_out);
            let max_y = max(y_in, y_out);
            
            if (min_y <= h_max && max_y >= h_min) {
                // overlap! figure out exact entry
                if y_in >= h_min && y_in <= h_max {
                    hit_d = current_t;
                } else if ray_dy < 0.0 && y_in > h_max {
                    hit_d = (h_max - origin.y) / ray_dy;
                } else if ray_dy > 0.0 && y_in < h_min {
                    hit_d = (h_min - origin.y) / ray_dy;
                } else {
                    hit_d = current_t; // fallback 
                }
                if (hit_d >= current_t && hit_d <= next_t) {
                    break;
                } else {
                    hit_d = 1000.0; // False intersection
                }
            }
            
            if t_max_x < t_max_z {
                current_t = t_max_x;
                t_max_x += t_delta_x;
                cur_tx = u32(i32(cur_tx) + step_x);
            } else {
                current_t = t_max_z;
                t_max_z += t_delta_z;
                cur_tz = u32(i32(cur_tz) + step_z);
            }
            
            if cur_tx < 0u || cur_tx >= uniforms.hmap_res.x || cur_tz < 0u || cur_tz >= uniforms.hmap_res.y {
                break;
            }
        }
        
        hits[probe_hit_base + r] = hit_d;
    }
}
