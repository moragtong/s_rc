use crate::cascades2_5d::Cascades2_5dPlugin;
mod bake_caps;
mod cascades2_5d;
mod heightmap;
mod heightmap_bake;
mod heightmap_verify;
mod heightmap_view;
mod hi_z;
mod hi_z_debug;
mod scene3d;
mod screenshot;
mod shade;
mod ui;

use bevy::camera::visibility::RenderLayers;
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass, NormalPrepass};
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy::render::renderer::RenderAdapterInfo;

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::gltf::GltfAssetLabel;
use heightmap::{DEBUG_LAYER, HeightmapPlugin, LIGHT_LAYER, SCENE_LAYER};
use heightmap_bake::HeightmapBakePlugin;
use heightmap_view::HeightmapViewPlugin;
use hi_z::{HiZ, HiZPlugin};
use hi_z_debug::HiZDebugPlugin;

fn main() {
    let measure_fps = std::env::var("S_RC_FPS").is_ok();
    let mut app = App::new();
    app.add_systems(Update, (draw_probes_gizmos, camera_readout))
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins((
            DefaultPlugins,
            HiZPlugin,
            HiZDebugPlugin,
            FreeCameraPlugin,
            HeightmapPlugin,
            HeightmapBakePlugin,
            heightmap_verify::HeightmapVerifyPlugin,
            HeightmapViewPlugin,
            ui::UiPlugin,
            Cascades2_5dPlugin,
            shade::ShadePlugin,
            screenshot::ScreenshotPlugin,
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
        ))
        .add_systems(Startup, (setup_scene, log_adapter_info));
    if measure_fps {
        app.add_systems(Update, log_fps);
    }
    app.run();
}

fn log_fps(mut frames: Local<Option<(std::time::Instant, u32)>>) {
    if frames.is_none() {
        *frames = Some((std::time::Instant::now(), 0));
    }
    let frames = frames.as_mut().unwrap();
    frames.1 += 1;
    let elapsed = frames.0.elapsed().as_secs_f32();
    if elapsed >= 2.0 {
        bevy::log::info!("FPS {:.1}", frames.1 as f32 / elapsed);
        frames.1 = 0;
        frames.0 = std::time::Instant::now();
    }
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    use bevy::world_serialization::WorldAssetRoot;
    let scene: Handle<bevy::world_serialization::WorldAsset> =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/Sponza/glTF/Sponza.gltf"));
    // The bake waits on this rather than guessing from a mesh count: until the
    // glTF and its dependencies have loaded there are no scene meshes at all,
    // and measuring then would bake an empty map.
    commands.insert_resource(crate::heightmap_bake::SceneAsset(scene.clone()));
    commands.spawn((WorldAssetRoot(scene), Transform::from_xyz(0.0, 0.0, 0.0)));

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        // Must include SCENE_LAYER: the shadow map renders its casters through
        // the light's own layers, and the glTF meshes are on the scene layer.
        // Restricting the light to LIGHT_LAYER alone silently empties the
        // shadow map.
        RenderLayers::from_layers(&[SCENE_LAYER, LIGHT_LAYER]),
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
        // The main camera sees the scene *and* the heightmap debug overlay;
        // the bake cameras see only the scene.
        RenderLayers::from_layers(&[SCENE_LAYER, DEBUG_LAYER, LIGHT_LAYER]),
        FreeCamera::default(),
        bevy::ui::IsDefaultUiCamera,
        initial_camera_transform(),
    ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::px(6.0, 6.0, 6.0, 6.0),
            ..default()
        },
        Text::new(""),
        TextColor(Color::WHITE),
        TextFont::from_font_size(16.0),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.65)),
        CameraReadout,
    ));
}

/// Prints the current viewpoint as an `S_RC_CAM` string, so a shot can be
/// reproduced exactly after flying the camera to the spot that matters.
fn camera_readout(
    camera: Query<&Transform, With<FreeCamera>>,
    mut readout: Query<&mut Text, With<CameraReadout>>,
    mut logged: Local<Option<(String, std::time::Instant)>>,
) {
    let Ok(transform) = camera.single() else {
        return;
    };
    let eye = transform.translation;
    let at = eye + transform.rotation * Vec3::NEG_Z * 10.0;
    let line = format!(
        "S_RC_CAM={:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
        eye.x, eye.y, eye.z, at.x, at.y, at.z
    );
    if let Ok(mut text) = readout.single_mut() {
        text.0 = line.clone();
    }
    let now = std::time::Instant::now();
    let changed = logged
        .as_ref()
        .map(|(prev, _)| *prev != line)
        .unwrap_or(true);
    let cooled = logged
        .as_ref()
        .map(|(_, at)| now.duration_since(*at) >= std::time::Duration::from_millis(250))
        .unwrap_or(true);
    if changed && cooled {
        info!("{line}");
        *logged = Some((line, now));
    }
}

/// Marks the `S_RC_CAM` readout in the corner.
#[derive(Component)]
struct CameraReadout;

/// `S_RC_CAM="ox,oy,oz,tx,ty,tz"` aims the camera somewhere specific, so a
/// shot can be taken from a known viewpoint instead of the default one. WASD
/// and QE still work afterwards.
fn initial_camera_transform() -> Transform {
    let fallback =
        || Transform::from_xyz(-9.0, 5.0, 5.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y);
    let Ok(spec) = std::env::var("S_RC_CAM") else {
        return fallback();
    };
    let nums: Vec<f32> = spec
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if nums.len() != 6 {
        warn!("S_RC_CAM needs 6 numbers (ox,oy,oz,tx,ty,tz); using default camera");
        return fallback();
    }
    let eye = Vec3::new(nums[0], nums[1], nums[2]);
    let at = Vec3::new(nums[3], nums[4], nums[5]);
    if eye.distance(at) < 1e-3 {
        warn!("S_RC_CAM eye and target coincide; using default camera");
        return fallback();
    }
    Transform::from_translation(eye).looking_at(at, Vec3::Y)
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

    let Some((_, cam_trans)) = camera_q.iter().next() else {
        return;
    };
    let cam_pos = cam_trans.translation();

    let hmap_min = hmap_res.min_bounds;
    let hmap_max = hmap_res.max_bounds;

    let cascades = crate::cascades2_5d::layout::Layout::new();

    for (i, c) in cascades.cascades.iter().enumerate() {
        let mut color = Color::srgb(1.0, 0.0, 0.0);
        if i == 0 {
            color = Color::srgb(0.0, 1.0, 0.0);
        }
        if i == 1 {
            color = Color::srgb(0.0, 0.5, 1.0);
        }
        if i == 2 {
            color = Color::srgb(0.5, 0.0, 1.0);
        }
        if i == 3 {
            color = Color::srgb(1.0, 1.0, 0.0);
        }

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
