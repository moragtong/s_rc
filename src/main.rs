mod hi_z;
mod hi_z_debug;

use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy::render::renderer::RenderAdapterInfo;
use hi_z::{HiZ, HiZPlugin};
use hi_z_debug::HiZDebugPlugin;

fn main() {
    App::new()
        // Deferred for every opaque material: GI needs a G-buffer to read
        // hit-point normals/albedo from, not just a lit color.
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins((DefaultPlugins, HiZPlugin, HiZDebugPlugin))
        .add_systems(Startup, (setup_scene, log_adapter_info))
        .run();
}

/// Ground plane + a cube + a directional "sun" light + camera.
/// The cube exists purely so there's something for the eventual
/// SDF march / shadow clipmap to have an opinion about later.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.35))),
        Transform::default(),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.2, 0.2))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    // Stands in for L_sun. Directional, shadows on, so we can confirm
    // Bevy's own shadow path works before we ever touch our clipmap.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.4, 0.0)),
    ));

    commands.spawn((
        Camera3d::default(),
        // Deferred can't do MSAA; set it explicitly rather than relying on
        // Bevy's warn-and-disable.
        Msaa::Off,
        DepthPrepass,
        DeferredPrepass,
        HiZ,
        Transform::from_xyz(4.0, 3.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Prints which WGPU backend and physical device got picked. On the
/// Iris Xe this should report the Vulkan (or DX12, on Windows) backend
/// and the Tiger Lake GT2 device name — confirms the environment is
/// actually talking to the iGPU we spent the last several turns
/// reasoning about, not silently falling back to a software rasterizer.
fn log_adapter_info(adapter_info: Res<RenderAdapterInfo>) {
    info!(
        "WGPU adapter: {} | type: {:?} | backend: {:?}",
        adapter_info.name, adapter_info.device_type, adapter_info.backend
    );
}
