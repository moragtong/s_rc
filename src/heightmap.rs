use bevy::{
    prelude::*,
    asset::RenderAssetUsages,
    render::{
        render_resource::*,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
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
    pub albedo_min: Handle<Image>,
    pub albedo_max: Handle<Image>,
    /// Sunlit outgoing radiance of the ceiling and floor surfaces, written
    /// on the GPU by the cascades' surface light pass.
    pub radiance_min: Handle<Image>,
    pub radiance_max: Handle<Image>,
    /// Highest ceiling or floor in the map; above it rays can't hit.
    pub top: f32,
}

/// Ceiling height of columns with nothing above the split (open sky).
pub const NO_CEILING: f32 = 1.0e4;

pub struct HeightmapPlugin;

impl Plugin for HeightmapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, setup_heightmap);
        app.add_plugins(ExtractResourcePlugin::<Heightmap>::default());
    }
}

fn create_empty_image(resolution: UVec2, format: TextureFormat, fill_size: usize, fill_val: f32) -> Image {
    let mut data = vec![0.0f32; (resolution.x * resolution.y) as usize * fill_size];
    data.fill(fill_val);
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
    let albedo_min = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    let albedo_max = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    let radiance_min = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    let radiance_max = images.add(create_empty_image(resolution, TextureFormat::Rgba32Float, 4, 0.0));
    
    commands.insert_resource(Heightmap {
        min_bounds, max_bounds, resolution,
        texture, metadata_min, metadata_max, albedo_min, albedo_max,
        radiance_min, radiance_max,
        top: 0.0,
    });
}


pub fn bake_dynamic_mesh(
    q_meshes: Query<(&bevy::prelude::Mesh3d, &GlobalTransform, Option<&MeshMaterial3d<StandardMaterial>>)>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut heightmap_res: ResMut<Heightmap>,
    mut frames: Local<u32>,
) {
    if q_meshes.iter().count() < 10 { return; }
    if *frames > 1 { return; }
    *frames += 1;
    if *frames != 1 { return; }
    
    if q_meshes.iter().count() == 0 { return; }
    
    let w = 512;
    let h = 512;
    let mut min_h = vec![1000.0f32; w * h];
    let mut max_h = vec![-1000.0f32; w * h];
    let mut meta_min = vec![Vec4::ZERO; w * h];
    let mut meta_max = vec![Vec4::ZERO; w * h];
    let mut alb_min = vec![Vec4::ZERO; w * h];
    let mut alb_max = vec![Vec4::ZERO; w * h];
    
    let bounds_min = Vec2::new(-20.0, -16.0);
    let bounds_max = Vec2::new(20.0, 16.0);
    let scale_x = w as f32 / (bounds_max.x - bounds_min.x);
    let scale_y = h as f32 / (bounds_max.y - bounds_min.y);
    
    let center_y = 5.0;
    
    for (mesh_handle, gtrans, mat_handle) in q_meshes.iter() {
        let mut albedo_base = Vec3::new(0.5, 0.5, 0.5);
        let mut tex_img: Option<&Image> = None;
        if let Some(mat_handle) = mat_handle {
            if let Some(mat) = materials.get(mat_handle) {
                let col = mat.base_color.to_linear();
                albedo_base = Vec3::new(col.red, col.green, col.blue);
                if let Some(tex) = mat.base_color_texture.as_ref() {
                    tex_img = images.get(tex);
                }
            }
        }
        
        let Some(mesh) = meshes.get(mesh_handle) else { continue; };
        let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|v| v.as_float3()) else { continue; };
        let Some(normals) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).and_then(|v| v.as_float3()) else { continue; };
        let uvs = if let Some(bevy::render::mesh::VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) { Some(uvs) } else { None };
        
        let indices: Vec<usize> = match mesh.indices() {
            Some(bevy::mesh::Indices::U16(i)) => i.iter().map(|&v| v as usize).collect(),
            Some(bevy::mesh::Indices::U32(i)) => i.iter().map(|&v| v as usize).collect(),
            None => (0..positions.len()).collect(),
        };
        
        let m = gtrans.compute_transform().to_matrix();
        let inv_m = m.inverse().transpose();
        
        for tri in indices.chunks(3) {
            if tri.len() < 3 { continue; }
            let mut v = [Vec3::ZERO; 3];
            let mut n = [Vec3::ZERO; 3];
            let mut uv_tri = [Vec2::ZERO; 3];
            for i in 0..3 {
                let pos = positions[tri[i]];
                let norm = normals[tri[i]];
                if let Some(uv_attr) = uvs { uv_tri[i] = Vec2::from(uv_attr[tri[i]]); }
                let wp = m * Vec4::new(pos[0], pos[1], pos[2], 1.0);
                v[i] = wp.truncate();
                let wn = (inv_m * Vec4::new(norm[0], norm[1], norm[2], 0.0)).truncate().normalize_or_zero();
                n[i] = wn;
            }
            
            let x_min = v[0].x.min(v[1].x).min(v[2].x);
            let x_max = v[0].x.max(v[1].x).max(v[2].x);
            let z_min = v[0].z.min(v[1].z).min(v[2].z);
            let z_max = v[0].z.max(v[1].z).max(v[2].z);
            
            let tx_min = ((x_min - bounds_min.x) * scale_x).floor() as i32;
            let tx_max = ((x_max - bounds_min.x) * scale_x).ceil() as i32;
            let ty_min = ((z_min - bounds_min.y) * scale_y).floor() as i32;
            let ty_max = ((z_max - bounds_min.y) * scale_y).ceil() as i32;
            
            let tx_min = tx_min.clamp(0, w as i32 - 1);
            let tx_max = tx_max.clamp(0, w as i32 - 1);
            let ty_min = ty_min.clamp(0, h as i32 - 1);
            let ty_max = ty_max.clamp(0, h as i32 - 1);
            
            for ty in ty_min..=ty_max {
                for tx in tx_min..=tx_max {
                    let px = (tx as f32 + 0.5) / scale_x + bounds_min.x;
                    let pz = (ty as f32 + 0.5) / scale_y + bounds_min.y;
                    let p = Vec2::new(px, pz);
                    
                    let a = Vec2::new(v[0].x, v[0].z);
                    let b = Vec2::new(v[1].x, v[1].z);
                    let c = Vec2::new(v[2].x, v[2].z);
                    
                    let denom = (b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y);
                    if denom.abs() < 1e-5 { continue; }
                    
                    let w0 = ((b.y - c.y)*(p.x - c.x) + (c.x - b.x)*(p.y - c.y)) / denom;
                    let w1 = ((c.y - a.y)*(p.x - c.x) + (a.x - c.x)*(p.y - c.y)) / denom;
                    let w2 = 1.0 - w0 - w1;
                    
                    if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                        let py = w0 * v[0].y + w1 * v[1].y + w2 * v[2].y;
                        let norm = (w0 * n[0] + w1 * n[1] + w2 * n[2]).normalize_or_zero();
                        let uv = w0 * uv_tri[0] + w1 * uv_tri[1] + w2 * uv_tri[2];
                        
                        let mut albedo = albedo_base;
                        if let Some(img) = tex_img {
                            let tw = img.texture_descriptor.size.width as usize;
                            let th = img.texture_descriptor.size.height as usize;
                            if tw > 0 && th > 0 {
                                let mut fx = uv.x.rem_euclid(1.0);
                                let mut fy = uv.y.rem_euclid(1.0);
                                let tx_uv = (fx * tw as f32) as usize;
                                let ty_uv = (fy * th as f32) as usize;
                                let idx = ((ty_uv % th) * tw + (tx_uv % tw)) * 4;
                                if idx + 2 < img.data.as_ref().unwrap().len() {
                                    let r = img.data.as_ref().unwrap()[idx] as f32 / 255.0;
                                    let g = img.data.as_ref().unwrap()[idx+1] as f32 / 255.0;
                                    let b = img.data.as_ref().unwrap()[idx+2] as f32 / 255.0;
                                    albedo = Vec3::new(r, g, b);
                                    // Linearize sRGB
                                    let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
                                    albedo = Vec3::new(lin(r), lin(g), lin(b));
                                }
                            }
                        }
                        
                        let idx = (ty as usize) * w + (tx as usize);
                        
                        if py <= center_y {
                            if py > max_h[idx] { 
                                max_h[idx] = py;
                                meta_max[idx] = Vec4::new(norm.x, norm.y, norm.z, 0.0);
                                alb_max[idx] = albedo.extend(0.0);
                            }
                        } else {
                            if py < min_h[idx] {
                                min_h[idx] = py;
                                meta_min[idx] = Vec4::new(norm.x, norm.y, norm.z, 0.0);
                                alb_min[idx] = albedo.extend(0.0);
                            }
                        }
                    }
                }
            }
        }
    }
    
    let mut top = f32::MIN;
    for idx in 0..w*h {
        if min_h[idx] == 1000.0 { min_h[idx] = NO_CEILING; } else { top = top.max(min_h[idx]); }
        if max_h[idx] == -1000.0 { max_h[idx] = center_y - 10.0; }
        top = top.max(max_h[idx]);
    }
    heightmap_res.top = top;
    
    if let Some(mut img) = images.get_mut(&heightmap_res.texture) {
        let mut data = Vec::with_capacity(w * h * 8);
        for i in 0..(w*h) {
            data.extend_from_slice(&min_h[i].to_ne_bytes());
            data.extend_from_slice(&max_h[i].to_ne_bytes());
        }
        img.data = Some(data);
    }
    
    for (img_handle, meta_arr) in [(&heightmap_res.metadata_min, &meta_min), (&heightmap_res.metadata_max, &meta_max), (&heightmap_res.albedo_min, &alb_min), (&heightmap_res.albedo_max, &alb_max)] {
        if let Some(mut img) = images.get_mut(img_handle) {
            let mut data = Vec::with_capacity(w * h * 16);
            for i in 0..(w*h) {
                let a = meta_arr[i].to_array();
                let mut bytes = [0u8; 16];
                for j in 0..4 { bytes[j*4..j*4+4].copy_from_slice(&a[j].to_ne_bytes()); }
                data.extend_from_slice(&bytes);
            }
            img.data = Some(data);
        }
    }
    println!("BAKED TRIANGLES WITH ALBEDO SAMPLING!");
}
