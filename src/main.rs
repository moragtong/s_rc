use crate::cascades2_5d::Cascades2_5dPlugin;
mod hi_z;
mod hi_z_debug;
mod cascades2_5d;
mod heightmap;
mod heightmap_view;
mod scene3d;
mod screenshot;
mod ui;
mod shade;

use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass, NormalPrepass};
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy::render::renderer::RenderAdapterInfo;

use hi_z::{HiZ, HiZPlugin};
use hi_z_debug::HiZDebugPlugin;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use heightmap::HeightmapPlugin;
use heightmap_view::HeightmapViewPlugin;
use bevy::gltf::GltfAssetLabel;

fn main() {
    App::new()
        .add_systems(Update, draw_probes_gizmos)
        .add_systems(Update, heightmap::bake_dynamic_mesh)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins((
            DefaultPlugins,
            HiZPlugin,
            HiZDebugPlugin,
            FreeCameraPlugin,
            HeightmapPlugin,
            HeightmapViewPlugin,
            ui::UiPlugin,
            Cascades2_5dPlugin,
            shade::ShadePlugin,
            screenshot::ScreenshotPlugin,
            
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
        ))
        .add_systems(Startup, (setup_scene, log_adapter_info))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    use bevy::world_serialization::WorldAssetRoot;
    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/Sponza/glTF/Sponza.gltf"))),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    commands.spawn((
        Camera3d::default(),
        // The indirect pass adds linear light before tonemapping.
        bevy::camera::Hdr,
        Msaa::Off,
        DepthPrepass,
        NormalPrepass,
        DeferredPrepass,
        HiZ,
        FreeCamera::default(),
        Transform::from_xyz(-9.0, 5.0, 5.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));
}

fn log_adapter_info(adapter_info: Res<RenderAdapterInfo>) {
    info!(
        "WGPU adapter: {} | type: {:?} | backend: {:?}",
        adapter_info.name, adapter_info.device_type, adapter_info.backend
    );
}

fn draw_probes_gizmos(
    mut gizmos: Gizmos,
    lighting_state: Res<crate::ui::LightingState>,
    hmap_res: Res<crate::heightmap::Heightmap>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
) {
    return;
    
    let Some((_, cam_trans)) = camera_q.iter().next() else { return; };
    let cam_pos = cam_trans.translation();
    
    let hmap_min = hmap_res.min_bounds;
    let hmap_max = hmap_res.max_bounds;
    
    let cascades = crate::cascades2_5d::layout::Layout::new();
    
    for (i, c) in cascades.cascades.iter().enumerate() {
        let mut color = Color::srgb(1.0, 0.0, 0.0);
        if i == 0 { color = Color::srgb(0.0, 1.0, 0.0); }
        if i == 1 { color = Color::srgb(0.0, 0.5, 1.0); }
        if i == 2 { color = Color::srgb(0.5, 0.0, 1.0); }
        if i == 3 { color = Color::srgb(1.0, 1.0, 0.0); }
        
        let radius = c.spacing * 0.15;
        let c_spacing = c.spacing;
        
        let s_x = (hmap_max.x - hmap_min.x) / c_spacing;
        let s_z = (hmap_max.y - hmap_min.y) / c_spacing;
        let limit_x = (s_x.ceil() as u32) + 1;
        let limit_z = (s_z.ceil() as u32) + 1;
        
        for z in 0..limit_z {
            for x in 0..limit_x {
                let w_pos_x = (x as f32 + 0.5) * c_spacing + hmap_min.x;
                let w_pos_z = (z as f32 + 0.5) * c_spacing + hmap_min.y;
                
                // Only draw spheres within 15 meters to save perf
                if (w_pos_x - cam_pos.x).abs() > 40.0 || (w_pos_z - cam_pos.z).abs() > 40.0 {
                    continue;
                }
                
                // For performance, we just draw them flat since reading GPU depth buffer continually requires readback callbacks.
                gizmos.sphere(Vec3::new(w_pos_x, 1.5, w_pos_z), radius, color);
            }
        }
    }
}
