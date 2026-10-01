use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;
use crate::heightmap::Heightmap;

pub fn bake_dynamic_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut q_meshes: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    mut heightmap_res: ResMut<Heightmap>,
    mut done: Local<bool>,
) {
    if *done { return; }
    
    // waiting for meshes to load...
    let mut baked_any = false;
    let mut bounds_min = vec2(-10.0, -10.0);
    let mut bounds_max = vec2(10.0, 10.0);
    let res = heightmap_res.resolution;
    
    // We need CPU access to meshes. The SceneRoot loads them, wait for Assets<Mesh>
    if q_meshes.iter().count() > 10 {
        // Tokyo has many meshes!
        // We will do a simple splat baker here if all meshes are loaded.
        let mut min_h = vec![-100.0f32; (res.x * res.y) as usize];
        let mut max_h = vec![-100.0f32; (res.x * res.y) as usize];
        
        for (ent, mesh_handle, gtrans) in q_meshes.iter() {
            if let Some(mesh) = meshes.get(mesh_handle) {
                baked_any = true;
                if let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                    for pos in positions {
                        // Project local to world
                        let world_pos = gtrans.compute_matrix() * Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        
                        let u = (world_pos.x - bounds_min.x) / (bounds_max.x - bounds_min.x);
                        let v = (world_pos.z - bounds_min.y) / (bounds_max.y - bounds_min.y);
                        
                        if u >= 0.0 && u < 1.0 && v >= 0.0 && v < 1.0 {
                            let tx = (u * res.x as f32) as usize;
                            let ty = (v * res.y as f32) as usize;
                            let idx = ty * res.x as usize + tx;
                            
                            max_h[idx] = max_h[idx].max(world_pos.y);
                            min_h[idx] = min_h[idx].max(world_pos.y); // just simplistic for now
                        }
                    }
                }
            } else {
                return; // not all loaded yet
            }
        }
        
        if baked_any {
            *done = true;
            info!("BAKED TOKYO DYNAMIC GLTF!");
            // upload to heightmap_res.texture? 
        }
    }
}
