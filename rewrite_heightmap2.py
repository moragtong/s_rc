import sys

code = """use bevy::{
    prelude::*,
    asset::RenderAssetUsages,
    render::{
        render_resource::*,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        RenderApp,
    },
};

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct Heightmap {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
    pub texture: Handle<Image>,
    pub metadata_min: Handle<Image>,
    pub metadata_max: Handle<Image>,
}

pub struct HeightmapPlugin;

impl Plugin for HeightmapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, setup_heightmap);
        app.add_plugins(ExtractResourcePlugin::<Heightmap>::default());
    }
}

fn create_empty_image(resolution: UVec2, format: TextureFormat, fill_size: usize, fill_val: f32) -> Image {
    let mut data = vec![0.0f32; (resolution.x * resolution.y) as usize * fill_size];
    for i in 0..data.len() {
        data[i] = fill_val;
    }
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let mut image = Image::new(
        Extent3d { width: resolution.x, height: resolution.y, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bytes,
        format,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image
}

fn setup_heightmap(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let resolution = UVec2::new(512, 512);
    let min_bounds = Vec2::new(-20.0, -16.0);
    let max_bounds = Vec2::new(20.0, 16.0);
    
    let texture = images.add(create_empty_image(resolution, TextureFormat::Rg32Float, 2, 0.0));
    let metadata_min = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    let metadata_max = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    
    commands.insert_resource(Heightmap {
        min_bounds, max_bounds, resolution,
        texture, metadata_min, metadata_max,
    });
}

pub fn bake_dynamic_mesh(
    q_meshes: Query<(&bevy::prelude::Mesh3d, &GlobalTransform, Option<&MeshMaterial3d<StandardMaterial>>)>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
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
        
        // Metadata: normal (xyz), albedo luminance (w)
        let mut meta_min = vec![Vec4::ZERO; (512 * 512) as usize];
        let mut meta_max = vec![Vec4::ZERO; (512 * 512) as usize];
        
        let bounds_min = Vec2::new(-20.0, -16.0);
        let bounds_max = Vec2::new(20.0, 16.0);
        
        for (mesh_handle, gtrans, mat_handle) in q_meshes.iter() {
            let mut albedo_lum = 0.5;
            if let Some(mat_handle) = mat_handle {
                if let Some(mat) = materials.get(mat_handle) {
                    let col = mat.base_color.to_linear();
                    albedo_lum = col.red * 0.2126 + col.green * 0.7152 + col.blue * 0.0722;
                }
            }
            
            if let Some(mesh) = meshes.get(mesh_handle) {
                let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|v| v.as_float3());
                let normals = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).and_then(|v| v.as_float3());
                
                if let (Some(positions), Some(normals)) = (positions, normals) {
                    for i in 0..positions.len() {
                        let pos = positions[i];
                        let norm = normals[i];
                        let world_pos = gtrans.compute_transform().to_matrix() * Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        let inv_trans = gtrans.compute_transform().to_matrix().inverse().transpose(); 
                        let world_norm = (inv_trans * Vec4::new(norm[0], norm[1], norm[2], 0.0)).truncate();
                        let world_norm = world_norm.normalize_or_zero();
                        
                        let u = (world_pos.x - bounds_min.x) / (bounds_max.x - bounds_min.x);
                        let v = (world_pos.z - bounds_min.y) / (bounds_max.y - bounds_min.y);
                        
                        if u >= 0.0 && u < 1.0 && v >= 0.0 && v < 1.0 {
                            let tx = (u * 512.0) as usize;
                            let ty = (v * 512.0) as usize;
                            let idx = ty * 512 + tx;
                            
                            if world_pos.y > max_h[idx] {
                                max_h[idx] = world_pos.y;
                                meta_max[idx] = Vec4::new(world_norm.x, world_norm.y, world_norm.z, albedo_lum);
                            }
                            if world_pos.y < min_h[idx] {
                                min_h[idx] = world_pos.y;
                                meta_min[idx] = Vec4::new(world_norm.x, world_norm.y, world_norm.z, albedo_lum);
                            }
                        }
                    }
                }
            }
        }
        
        if let Some(mut img) = images.get_mut(&heightmap_res.texture) {
            let mut data = Vec::with_capacity(512 * 512 * 8);
            for i in 0..(512*512) {
                let mn = if min_h[i] == 1000.0 { 0.0 } else { min_h[i] };
                let mx = if max_h[i] == -1000.0 { 0.0 } else { max_h[i] };
                data.extend_from_slice(&mn.to_ne_bytes());
                data.extend_from_slice(&mx.to_ne_bytes());
            }
            img.data = Some(data);
        }
        
        if let Some(mut img) = images.get_mut(&heightmap_res.metadata_min) {
            let mut data = Vec::with_capacity(512 * 512 * 16);
            for i in 0..(512*512) {
                let a = meta_min[i].to_array(); 
                let mut bytes = [0u8; 16]; 
                for j in 0..4 { bytes[j*4..j*4+4].copy_from_slice(&a[j].to_ne_bytes()); }
                data.extend_from_slice(&bytes);
            }
            img.data = Some(data);
        }
        
        if let Some(mut img) = images.get_mut(&heightmap_res.metadata_max) {
            let mut data = Vec::with_capacity(512 * 512 * 16);
            for i in 0..(512*512) {
                let a = meta_max[i].to_array(); 
                let mut bytes = [0u8; 16]; 
                for j in 0..4 { bytes[j*4..j*4+4].copy_from_slice(&a[j].to_ne_bytes()); }
                data.extend_from_slice(&bytes);
            }
            img.data = Some(data);
        }
        
        println!("BAKED SCENE GLTF INTO HEIGHTMAP WITH METADATA! Meshes count: {}", q_meshes.iter().count());
    }
}
"""

with open("src/heightmap.rs", "w") as f:
    f.write(code)

