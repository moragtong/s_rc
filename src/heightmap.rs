use bevy::{
    prelude::*,
    asset::RenderAssetUsages,
    render::{
        render_resource::*,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        RenderApp,
    },
};
use crate::scene3d::Scene3d;

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct Heightmap {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
    pub texture: Handle<Image>,
}

pub struct HeightmapPlugin;

impl Plugin for HeightmapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, setup_heightmap);
        
        app.add_plugins(ExtractResourcePlugin::<Heightmap>::default());
    }
}

fn setup_heightmap(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let resolution = UVec2::new(512, 512);
    let min_bounds = Vec2::new(-8.0, -8.0);
    let max_bounds = Vec2::new(8.0, 8.0);
    
    // Create the image
    let mut data = vec![0.0f32; (resolution.x * resolution.y * 2) as usize];
    let scene = Scene3d::room(0.2);
    
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
    }
    
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    
    let mut image = Image::new(
        Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rg32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= (TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING);
    
    let texture = images.add(image);
    
    commands.insert_resource(Heightmap {
        min_bounds,
        max_bounds,
        resolution,
        texture,
    });
}

#[derive(bevy::render::render_resource::ShaderType, Clone, Copy, Default)]
pub struct HeightmapUniform {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
}

pub fn bake_dynamic_mesh(
    mut commands: Commands,
    mut q_meshes: Query<(&bevy::prelude::Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    heightmap_res: Res<Heightmap>,
    mut done: Local<bool>,
) {
    if *done { return; }
    
    // waiting for meshes to load...
    if q_meshes.iter().count() > 10 {
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
                return; // wait for all
            }
        }
        
        // Write to image
        if let Some(mut img) = images.get_mut(&heightmap_res.texture) {
            let mut data = Vec::with_capacity(512 * 512 * 8);
            for i in 0..(512*512) {
                let mn = if min_h[i] == 1000.0 { -2.0 } else { min_h[i] };
                let mx = if max_h[i] == -1000.0 { -2.0 } else { max_h[i] };
                data.extend_from_slice(&mn.to_ne_bytes());
                data.extend_from_slice(&mx.to_ne_bytes());
            }
            img.data = Some(data);
        }
        
        *done = true;
        println!("BAKED TOKYO GLTF INTO HEIGHTMAP!");
    }
}
