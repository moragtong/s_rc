import re

with open("src/heightmap.rs", "r") as f:
    text = f.read()

# Setup heightmap: remove Scene3d::room
old_setup = """    let scene = Scene3d::room(0.2);
    
    for y in 0..resolution.y {
        for x in 0..resolution.x {
            let px = min_bounds.x + (x as f32 + 0.5) / resolution.x as f32 * (max_bounds.x - min_bounds.x);
            let pz = min_bounds.y + (y as f32 + 0.5) / resolution.y as f32 * (max_bounds.y - min_bounds.y);
            
            let mut min_h = f32::INFINITY;
            let mut max_h = f32::NEG_INFINITY;
            
            for b in &scene.boxes {
                if (px - b.center.x).abs() <= b.half_size.x && (pz - b.center.z).abs() <= b.half_size.z {
                    let top = b.center.y + b.half_size.y;
                    let bottom = b.center.y - b.half_size.y;
                    if bottom < min_h { min_h = bottom; }
                    if top > max_h { max_h = top; }
                }
            }
            
            if min_h == f32::INFINITY {
                min_h = 0.0;
                max_h = 0.0; // fallback if outside
            }
            
            let idx = ((y * resolution.x + x) * 2) as usize;
            data[idx] = min_h;
            data[idx + 1] = max_h;
        }
    }"""

new_setup = """    for i in 0..(resolution.x * resolution.y) as usize {
        data[i * 2] = -100.0;
        data[i * 2 + 1] = -100.0;
    }"""

text = text.replace(old_setup, new_setup)

# Update bake_dynamic_mesh
bake_regex = r"pub fn bake_dynamic_mesh.*?done = true;\s+println!\(\"BAKED TOKYO GLTF INTO HEIGHTMAP!\"\);\s+\}\s+\}"

new_bake = """pub fn bake_dynamic_mesh(
    mut commands: Commands,
    mut q_meshes: Query<(&bevy::prelude::Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    heightmap_res: Res<Heightmap>,
    mut frames: Local<u32>,
) {
    if *frames > 150 { return; }
    *frames += 1;
    if *frames != 150 { return; }
    
    if q_meshes.iter().count() > 0 {
        let mut min_h = vec![1000.0f32; (512 * 512) as usize];
        let mut max_h = vec![-1000.0f32; (512 * 512) as usize];
        
        let bounds_min = Vec2::new(-8.0, -8.0);
        let bounds_max = Vec2::new(8.0, 8.0);
        
        for (mesh_handle, gtrans) in q_meshes.iter() {
            if let Some(mesh) = meshes.get(mesh_handle) {
                if let Some(bevy::render::mesh::VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                    for pos in positions {
                        let world_pos = gtrans.to_matrix() * Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        let u = (world_pos.x - bounds_min.x) / (bounds_max.x - bounds_min.x);
                        let v = (world_pos.z - bounds_min.y) / (bounds_max.y - bounds_min.y);
                        
                        if u >= 0.0 && u < 1.0 && v >= 0.0 && v < 1.0 {
                            let tx = (u * 512.0) as usize;
                            let ty = (v * 512.0) as usize;
                            let idx = ty * 512 + tx;
                            
                            max_h[idx] = max_h[idx].max(world_pos.y);
                            min_h[idx] = min_h[idx].min(world_pos.y);
                        }
                    }
                }
            } else {
                *frames -= 1; // Try again next frame if not all meshes are loaded
                return; 
            }
        }
        if let Some(mut img) = images.get_mut(&heightmap_res.texture) {
            let mut data = Vec::with_capacity(512 * 512 * 8);
            for i in 0..(512*512) {
                let mn = if min_h[i] == 1000.0 { -10.0 } else { min_h[i] };
                let mx = if max_h[i] == -1000.0 { -10.0 } else { max_h[i] };
                data.extend_from_slice(&mn.to_ne_bytes());
                data.extend_from_slice(&mx.to_ne_bytes());
            }
            img.data = Some(data);
        }
        
        println!("BAKED SCENE GLTF INTO HEIGHTMAP! Meshes count: {}", q_meshes.iter().count());
    }
}"""

text = re.sub(bake_regex, new_bake, text, flags=re.DOTALL)

with open("src/heightmap.rs", "w") as f:
    f.write(text)
