mod hi_z;
mod hi_z_debug;
mod cascades3d;
mod clipmap;
mod scene3d;
mod sdf_bake;
mod sdf_view;

use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy::render::renderer::RenderAdapterInfo;
use hi_z::{HiZ, HiZPlugin};
use hi_z_debug::HiZDebugPlugin;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use cascades3d::Cascades3dPlugin;
use clipmap::{ClipmapCamera, ClipmapPlugin};
use scene3d::Scene3d;
use sdf_view::{SceneResource, SdfViewPlugin};

fn main() {
    App::new()
        // Deferred for every opaque material: GI needs a G-buffer to read
        // hit-point normals/albedo from, not just a lit color.
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins((
            DefaultPlugins,
            HiZPlugin,
            HiZDebugPlugin,
            FreeCameraPlugin,
            ClipmapPlugin,
            SdfViewPlugin,
            Cascades3dPlugin,
            // GPU timestamp spans for every pass (render/<span>/elapsed_gpu).
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
        ))
        .add_systems(Startup, (setup_scene, log_adapter_info))
        .run();
}

/// The analytic test room (see `scene3d`), and a free camera outside the
/// doorway that the SDF clipmap follows. The room's walls are 20 cm, 1.6
/// finest-tier voxels; thinner walls are a separate test.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let scene = Scene3d::room(0.2);
    scene3d::spawn(&scene, &mut commands, &mut meshes, &mut materials);
    commands.insert_resource(clipmap::SunLight {
        toward: scene.sun.toward,
        irradiance: scene.sun.irradiance,
        k: 1.0 / scene.sun.angular_radius.tan(),
    });
    commands.insert_resource(SceneResource(scene));

    commands.spawn((
        Camera3d::default(),
        // Deferred can't do MSAA; set it explicitly rather than relying on
        // Bevy's warn-and-disable.
        Msaa::Off,
        DepthPrepass,
        DeferredPrepass,
        HiZ,
        FreeCamera::default(),
        ClipmapCamera,
        Transform::from_xyz(-9.0, 1.7, 2.0).looking_at(Vec3::new(0.0, 1.2, -0.5), Vec3::Y),
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
