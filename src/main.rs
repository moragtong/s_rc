use crate::cascades2_5d::Cascades2_5dPlugin;
mod hi_z;
mod hi_z_debug;
mod cascades2_5d;
mod heightmap;
mod heightmap_view;
mod scene3d;

use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
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
        .add_systems(Update, heightmap::bake_dynamic_mesh)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins((
            DefaultPlugins,
            HiZPlugin,
            HiZDebugPlugin,
            FreeCameraPlugin,
            HeightmapPlugin,
            HeightmapViewPlugin,
            Cascades2_5dPlugin,
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
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/Fox.glb"))),
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
        Msaa::Off,
        DepthPrepass,
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
