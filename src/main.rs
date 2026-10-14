use crate::cascades2_5d::Cascades2_5dPlugin;
pub use crate::cascades2_5d::LOWPASS_BLUR_PASSES;
mod bake_camera_readback;
mod cascades2_5d;
mod heightmap;
mod heightmap_bake;
mod heightmap_verify;
mod heightmap_view;
mod probe_overlay;
mod screenshot;
mod shade;
mod ui;
mod winding;

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

fn main() {
    let measure_fps = std::env::var("S_RC_FPS").is_ok();
    let mut app = App::new();
    app.add_systems(
        Update,
        (
            draw_probes_gizmos,
            camera_readout,
            animate_sun,
            sync_sky_color,
        ),
    )
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .add_plugins((
        DefaultPlugins,
        FreeCameraPlugin,
        HeightmapPlugin,
        HeightmapBakePlugin,
        winding::WindingPlugin,
        heightmap_verify::HeightmapVerifyPlugin,
        HeightmapViewPlugin,
        ui::UiPlugin,
        Cascades2_5dPlugin,
        shade::ShadePlugin,
        probe_overlay::ProbeOverlayPlugin,
        screenshot::ScreenshotPlugin,
        bevy::render::diagnostic::RenderDiagnosticsPlugin,
        bake_camera_readback::BakeCameraReadbackPlugin,
    ))
    // Bevy's `LightPlugin` inserts `GlobalAmbientLight` at brightness 80
    // cd/m^2. This scene lights the room entirely with the baked cascade
    // probes (plus the sun), so that flat ambient is a second, unrelated
    // light that washes the room out. `NONE` is the same resource at
    // brightness 0. Delete this line to get Bevy's default ambient back.
    .insert_resource(bevy::light::GlobalAmbientLight::NONE)
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

/// `S_RC_SUN=azimuth,elevation` in degrees: azimuth about world +Y measured
/// from +Z, elevation above the horizon. Pins the sun so a state can be
/// reproduced and A/B'd, which the 20 s sweep cannot be; the default matches
/// `Transform::from_xyz(4, 8, 4).looking_at(ZERO, Y)`, az 45 el 54.7. Pair
/// with `S_RC_SUN_STILL` to stop it sweeping off that mark.
fn sun_transform() -> Transform {
    let mut transform = Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y);
    let Ok(value) = std::env::var("S_RC_SUN") else {
        return transform;
    };
    let mut parts = value.split(',');
    let parse = |part: Option<&str>, fallback: f32| {
        part.and_then(|p| p.trim().parse::<f32>().ok())
            .unwrap_or(fallback)
    };
    let azimuth = parse(parts.next(), 45.0).to_radians();
    let elevation = parse(parts.next(), 54.7).to_radians();
    // The light shines along -Z of its transform, so -Z becomes the direction
    // of travel: down at `elevation`, swung round by `azimuth`. The sun the
    // shaders use is the negation of that.
    transform.rotation = Quat::from_rotation_y(azimuth) * Quat::from_rotation_x(-elevation);
    transform
}

/// One turn of the sun, in seconds. Fast enough that the four-frame relight
/// interval shows: the direct light and its shadows move every frame while the
/// probes catch up over the interval, which is the whole point of the
/// cross-fade.
const SUN_SWEEP_PERIOD_SECS: f32 = 20.0;

/// Sweeps the sun so the lag between direct and indirect lighting is visible.
/// `S_RC_SUN_STILL` freezes it, for any shot that has to be reproducible.
fn animate_sun(
    time: Res<Time>,
    mut frozen: Local<Option<bool>>,
    mut printed: Local<Option<f32>>,
    mut lights: Query<&mut Transform, With<DirectionalLight>>,
) {
    if frozen.is_none() {
        *frozen = Some(std::env::var("S_RC_SUN_STILL").is_ok());
    }
    // Print the starting angle once on frame 0, before anything moves.
    let print_now = printed.is_none();
    if !frozen.unwrap_or(false) {
        let period = SUN_SWEEP_PERIOD_SECS;
        let delta = time.delta_secs() * (std::f32::consts::TAU / period);
        for mut transform in &mut lights {
            transform.rotate_y(delta);
        }
    }
    // Direction the light is shining *toward* (its -Z).
    let toward = lights
        .iter()
        .next()
        .map(|t| -t.forward().as_vec3())
        .unwrap_or(Vec3::Y);
    // Recover azimuth and elevation from the unit vector so the log line is in
    // the same format as S_RC_SUN.
    let elevation = toward.y.asin().to_degrees();
    let azimuth = toward.x.atan2(toward.z).to_degrees().rem_euclid(360.0);
    // Print on any change, throttled to 1 Hz so the log stays readable.
    let print_cooled = printed.map(|p| (azimuth - p).abs() >= 18.0).unwrap_or(true);
    if print_now || print_cooled {
        info!("S_RC_SUN={azimuth:.0},{elevation:.1}");
        *printed = Some(azimuth);
    }
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    use bevy::world_serialization::WorldAssetRoot;
    let scene: Handle<bevy::world_serialization::WorldAsset> =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/room.glb"));
    // The bake waits on this rather than guessing from a mesh count: until the
    // glTF and its dependencies have loaded there are no scene meshes at all,
    // and measuring then would bake an empty map.
    commands.insert_resource(crate::heightmap_bake::SceneAsset(scene.clone()));
    commands.spawn((WorldAssetRoot(scene), Transform::from_xyz(0.0, 0.0, 0.0)));

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: false,
            ..default()
        },
        // Must include SCENE_LAYER: the shadow map renders its casters through
        // the light's own layers, and the glTF meshes are on the scene layer.
        // Restricting the light to LIGHT_LAYER alone silently empties the
        // shadow map.
        RenderLayers::layer(LIGHT_LAYER),
        sun_transform(),
    ));

    let camera = commands
        .spawn((
            Camera3d::default(),
            // The indirect pass adds linear light before tonemapping.
            bevy::camera::Hdr,
            Msaa::Off,
            DepthPrepass,
            NormalPrepass,
            DeferredPrepass,
            // The main camera sees the scene *and* the heightmap debug overlay;
            // the bake cameras see only the scene.
            RenderLayers::from_layers(&[SCENE_LAYER, DEBUG_LAYER, LIGHT_LAYER]),
            FreeCamera::default(),
            bevy::ui::IsDefaultUiCamera,
            initial_camera_transform(),
        ))
        .id();

    // SSAO runs in the deferred prepass and multiplies ambient light, but with
    // ambient off it has little to darken. Gated OFF-by-`S_RC_SSAO=0` so an A/B
    // can be shot with one binary.
    if std::env::var("S_RC_SSAO").as_deref() != Ok("0") {
        commands
            .entity(camera)
            .insert(bevy::pbr::ScreenSpaceAmbientOcclusion::default());
    }

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            padding: UiRect::px(6.0, 6.0, 6.0, 6.0),
            border_radius: BorderRadius::all(Val::Px(4.0)),
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

/// Paints the main camera's backdrop with the same sky the cascade probes
/// record: `cascades2_5d::sky_color` is the radiance a probe ray stores on
/// a miss, so the sky showing through the doorway should read as that exact
/// sky rather than Bevy's default grey clear. The irradiance here is the
/// same `light.color * illuminance * exposure` as `cascades2_5d::update_sun`
/// (`S_RC_SKY` overrides both, because both go through `sky_color`).
fn sync_sky_color(
    mut cameras: Query<(&mut Camera, Option<&bevy::camera::Exposure>), With<FreeCamera>>,
    lights: Query<(&DirectionalLight, &GlobalTransform)>,
) {
    let (mut camera, exposure) = match cameras.single_mut() {
        Ok(c) => c,
        Err(_) => return,
    };
    let Ok((light, _transform)) = lights.single() else {
        return;
    };
    let exposure = exposure.map(|e| e.exposure()).unwrap_or(1.0);
    let color = light.color.to_linear();
    let irradiance = Vec3::new(color.red, color.green, color.blue) * light.illuminance * exposure;
    let sky = crate::cascades2_5d::sky_color(irradiance);
    camera.clear_color = ClearColorConfig::Custom(Color::linear_rgb(sky.x, sky.y, sky.z));
}

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
    mut _gizmos: Gizmos,
    _lighting_state: Res<crate::ui::LightingState>,
    _hmap_res: Res<crate::heightmap::Heightmap>,
    _camera_q: Query<(&Camera, &GlobalTransform)>,
) {
    return;
}
