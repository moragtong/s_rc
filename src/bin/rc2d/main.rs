//! 2D radiance cascades testbed (roadmap step 2).
//!
//! Runs flatland RC and a brute-force reference over the same analytic
//! scene, and logs the relative RMSE between them every couple of seconds.
//! The point is a number to tune against, not an eyeball.
//!
//! Controls:
//! - `1` / `2` / `3`: show cascades / reference / |difference|
//! - `B`: toggle plain bilinear merge vs. bilinear fix
//! - `Up` / `Down`: double / halve the cascade-0 interval scale `b`
//! - `P`: toggle cascade-0 probe spacing s0 between 1 and 2 px
//! - `D`: cycle how many of the lowest cascades use the per-direction
//!   merge (0-3, then all; see the cascade shader)
//! - `S`: switch scene: lights and occluders / room with thin diffuse walls,
//!   a spotlight outside its only entrance and lights around its outside
//!   (resets the light position)
//! - `G`: cycle diffuse bounces 0-4 (room scene; restarts the reference)
//! - `C`: run the intermediate bounce passes at double probe spacing
//!   (they only feed wall lighting); the final, visible pass is unchanged
//! - `J`: diagnostic: rotate all ray directions by a different angle each
//!   frame and average the result over frames. Error that goes away is
//!   aliasing from the fixed ray directions; error that stays is bias.
//! - `R`: restart reference accumulation
//! - `Space`: pause / resume the reference. Once it has converged, pausing
//!   frees its GPU time (~50 ms/frame in the room scene) for smooth viewing;
//!   metrics keep comparing against the frozen result. Anything that
//!   restarts the reference also resumes it, so it can't silently go stale.
//! - Left-drag: move the warm light (restarts the reference)
//!
//! Every 2s it also logs per-stage GPU time from timestamp queries, when the
//! adapter supports timestamps inside compute passes.
//!
//! The reference converges over time (noise ~ 1/sqrt(rays)); RMSE figures
//! are only trustworthy once its ray count is in the thousands.
//!
//! Measured on this scene at 512x512, reference >40k rays/px:
//!
//! | b (px) | plain merge | bilinear fix |
//! |--------|-------------|--------------|
//! | 0.25   | 8.54%       | 6.37%        |
//! | 0.5    | 5.63%       | 4.19%        |
//! | 1      | 6.42%       | 4.93%        |
//! | 2      | 9.94%       | 8.42%        |
//!
//! Plain merge leaks warm light through the thin wall and rings at interval
//! boundaries; the bilinear fix removes the leak and most of the ringing.
//! What's left is streaking along shadow edges (angular resolution).
//!
//! (Those errors are at s0 = 1 px.) The fix's 4x traces cost about the same
//! as the plain merge: with vec4<f32> cascades, both ran ~19-24 ms on Iris
//! Xe, and cascade 0 alone took ~3 ms despite 2px rays. It's memory-bound,
//! not trace-bound.
//!
//! Bandwidth cuts, bilinear fix on, same-run comparisons:
//!
//! | cascade storage | s0   | best b | cascades GPU | rel. RMSE |
//! |-----------------|------|--------|--------------|-----------|
//! | rgba32f         | 1 px | 0.5    | ~23 ms       | 4.37%     |
//! | rgba16f packed  | 1 px | 0.5    | ~15 ms       | 4.37%     |
//! | rgba16f packed  | 2 px | 1      | ~4 ms        | 6.29%     |
//!
//! fp16 is free accuracy-wise. s0 = 2 is ~4x cheaper. At s0 = 2 the resolve
//! must skip cascade-0 probes that sit inside shapes, or every shape edge
//! gets a halo (12.48% without).
//!
//! The remaining near-light error was a bias, not aliasing: rotating all
//! ray directions per frame and averaging (`J`) barely changed it. The
//! cascades were too bright within a few px of the light's surface, from
//! tracing each arc's near segment along its center only (see the cascade
//! shader). Merging per child direction in the lowest cascades (`D`):
//!
//! | s0   | b   | per-direction levels | cascades GPU | rel. RMSE | 0-2px from light |
//! |------|-----|----------------------|--------------|-----------|------------------|
//! | 1 px | 0.5 | 0                    | ~14 ms       | 4.18%     | 16.1%            |
//! | 1 px | 0.5 | 2                    | ~20 ms       | 2.74%     | 8.1%             |
//! | 1 px | 0.5 | all                  | ~49 ms       | 2.64%     | 8.4%             |
//! | 2 px | 1   | 0                    | ~3.6 ms      | 6.28%     | 19.1%            |
//! | 2 px | 1   | 2 (default)          | ~5.5 ms      | 3.61%     | 7.2%             |
//! | 2 px | 1   | 3                    | ~7.8 ms      | 3.50%     | 7.3%             |
//!
//! Higher levels cost the most (long rays x 4 traces) and help the least.
//!
//! Room scene (`S`), 2 diffuse bounces, same-run comparisons at ~12-17k
//! reference rays/px. Path-traced bounces make the reference noisier, so
//! the in-room RMSE includes reference noise. The in-room brightness sums
//! average that noise out, so they're the reliable leak signal:
//!
//! | merge        | s0   | b   | levels | rel. RMSE | in-room RMSE | in-room brightness |
//! |--------------|------|-----|--------|-----------|--------------|--------------------|
//! | plain        | 2 px | 1   | 0      | 13.35%    | 49.66%       | +60.1%             |
//! | bilinear fix | 1 px | 0.5 | 2      | 7.67%     | 11.77%       | -0.7%              |
//! | bilinear fix | 2 px | 1   | 2      | 7.69%     | 15.75%       | -1.8%              |
//!
//! (After moving the bounce lookup half a probe spacing off the surface,
//! see bounce_radiance: default in-room brightness -1.2%. Wall pixels now
//! show their nearest face's lit color; wall faces: 26.4% rel. RMSE, -6.4%
//! brightness vs reference.)
//!
//! The plain merge leaks the outside lights' color through the 2px walls
//! onto their inner faces, and that leaked light then bounces around the
//! room. With the bilinear fix and side-of-surface bounce lookup, no
//! outside color shows up inside. What's left is blurred, blotchy edges on
//! the spotlight's cone and beam, a sharp angular feature the cascades'
//! fixed directions resolve poorly.
//!
//! Cost, room at 2 bounces (3 cascade passes), same run: the per-pass cost
//! is ~1.3x scene 0's (9.2 vs 7.1 ms direct), plus ~3 ms per bounce pass
//! for the lookups. Most of the total is the pass count. Running the
//! intermediate passes at double probe spacing (`C`, default on) cut
//! cascade work 37.1 -> 24.5 ms with every error figure within the spread
//! of the off runs (in-room 16.2% vs 15.7-16.5%, wall faces -6.3% vs
//! -6.4%). Compiling the bounce lookup out of the direct pass was also
//! tried: ~2-3%, within noise, so not kept. The path-traced reference, at
//! ~53 ms, is the largest cost in the room scene.
//! A direction-first layout wasn't tried: its payoff is hardware-filtered
//! probe blending, which the bilinear fix can't use (each neighbour is
//! merged with its own near segment before weighting).

use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{DiagnosticPath, DiagnosticsStore},
    core_pipeline::{Core2d, Core2dSystems, tonemapping::Tonemapping},
    image::ImageSampler,
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        diagnostic::{RecordDiagnostics, RenderDiagnosticsPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
    window::WindowResolution,
};

mod room;

const SIZE: u32 = 512;
const _: () = assert!(SIZE.is_power_of_two(), "probe grids assume exact halving");
const SHADER_PATH: &str = "shaders/rc2d.wgsl";
/// Imported by the main shader; must stay loaded for its imports to resolve.
const SHADER_LIBRARIES: [&str; 2] = ["shaders/rc2d_sdf.wgsl", room::SHADER_PATH];
/// Every cascade stores (W/(s0 2^n))(H/(s0 2^n)) probes * 4^(n+1) rays =
/// 4WH/s0^2 entries. Buffers are sized for s0 = 1, the largest case.
const CASCADE_ENTRIES: u64 = 4 * SIZE as u64 * SIZE as u64;
const PIXELS: u64 = SIZE as u64 * SIZE as u64;
const VEC4: u64 = 16;
/// rgba16float packed as two u32s.
const CASCADE_ENTRY_BYTES: u64 = 8;
const MAX_CASCADES: u32 = 9;
/// Two vec4<f32> per pixel; layout in the shader's `error` binding.
const ERROR_BYTES_PER_PIXEL: u64 = 32;
fn scene_light_start(scene: u32) -> Vec2 {
    match scene {
        room::SCENE_ID => room::LIGHT_START,
        _ => Vec2::new(110.0, 400.0),
    }
}

#[derive(Resource)]
struct ShaderLibraries(#[allow(dead_code)] Vec<Handle<Shader>>);

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "rc2d".into(),
                    // 1:1 canvas pixels to screen pixels, regardless of DPI.
                    resolution: WindowResolution::new(SIZE, SIZE).with_scale_factor_override(1.0),
                    resizable: false,
                    ..default()
                }),
                ..default()
            }),
            Rc2dPlugin,
        ))
        .run();
}

struct Rc2dPlugin;

#[derive(Resource, Clone, Copy, ExtractResource)]
struct Rc2dSettings {
    view_mode: u32,
    bilinear_fix: bool,
    base_interval: f32,
    ref_frame: u32,
    /// Makes the next extracted frame overwrite the reference instead of
    /// accumulating onto it.
    reset_reference: bool,
    /// Skip reference accumulation; see the `Space` control.
    reference_paused: bool,
    ref_rays: u32,
    exposure: f32,
    diff_gain: f32,
    light_pos: Vec2,
    /// log2 of the cascade-0 probe spacing s0 in px.
    probe0_log2: u32,
    jitter: bool,
    /// Cascades below this use the per-direction merge.
    per_direction_levels: u32,
    scene: u32,
    /// Diffuse bounces in the room scene (scene 0 has no diffuse surfaces).
    bounces: u32,
    coarse_bounce_passes: bool,
    /// Frames averaged into the jittered history; like `ref_frame`, 0 means
    /// overwrite. Reset whenever anything the cascades depend on changes.
    rc_frame: u32,
}

impl Default for Rc2dSettings {
    fn default() -> Self {
        Self {
            view_mode: 0,
            bilinear_fix: true,
            // With s0 = 2 px: cascade 0 covers [0, 4px]. Measured best at
            // this spacing; see the module docs.
            base_interval: 1.0,
            ref_frame: 0,
            reset_reference: true,
            reference_paused: false,
            ref_rays: 16,
            exposure: 2.0,
            diff_gain: 8.0,
            light_pos: scene_light_start(0),
            probe0_log2: 1,
            jitter: false,
            per_direction_levels: 2,
            scene: 0,
            bounces: 2,
            // Measured free accuracy-wise; see the module docs.
            coarse_bounce_passes: true,
            rc_frame: 0,
        }
    }
}

type CascadeKey = (u32, bool, u32, u32, [u32; 2], bool, u32, u32, bool);

impl Rc2dSettings {
    /// Probe spacing (log2) and interval scale b for one cascade pass. With
    /// coarse bounce passes, the passes before the last run at double
    /// spacing and double b, keeping the b-to-spacing ratio measured best.
    fn pass_layout(&self, iteration: u32) -> (u32, f32) {
        let last = iteration == self.effective_bounces();
        if self.coarse_bounce_passes && !last {
            (self.probe0_log2 + 1, self.base_interval * 2.0)
        } else {
            (self.probe0_log2, self.base_interval)
        }
    }

    fn effective_bounces(&self) -> u32 {
        if self.scene == room::SCENE_ID { self.bounces } else { 0 }
    }

    /// Everything the cascades' output depends on; a change restarts the
    /// jittered average.
    fn cascade_key(&self) -> CascadeKey {
        (
            self.base_interval.to_bits(),
            self.bilinear_fix,
            self.per_direction_levels,
            self.probe0_log2,
            [self.light_pos.x.to_bits(), self.light_pos.y.to_bits()],
            self.jitter,
            self.scene,
            self.effective_bounces(),
            self.coarse_bounce_passes,
        )
    }

    /// Per-frame rotation for the jitter diagnostic: a golden-ratio sequence
    /// over [0, pi/2), cascade 0's ray spacing. Rotating by that full range
    /// rotates every cascade's direction set uniformly.
    fn ray_rotation(&self) -> f32 {
        if !self.jitter {
            return 0.0;
        }
        (self.rc_frame as f32 * 0.618_034).fract() * std::f32::consts::FRAC_PI_2
    }

    /// Fewest cascades whose last interval end, b*4^N, reaches the canvas
    /// diagonal, so every ray can see the whole scene.
    fn cascade_count(&self) -> u32 {
        cascade_count_for(self.base_interval)
    }
}

fn cascade_count_for(base_interval: f32) -> u32 {
    let diagonal = (2.0 * (SIZE as f32).powi(2)).sqrt();
    (1..=MAX_CASCADES)
        .find(|&n| base_interval * 4f32.powi(n as i32) >= diagonal)
        .unwrap_or(MAX_CASCADES)
}

impl Rc2dSettings {
}

#[derive(Resource, Clone, ExtractResource)]
struct Rc2dBuffers {
    cascade_a: Handle<ShaderBuffer>,
    cascade_b: Handle<ShaderBuffer>,
    reference: Handle<ShaderBuffer>,
    error: Handle<ShaderBuffer>,
    rc_history: Handle<ShaderBuffer>,
    /// Cascade 0 of the previous bounce pass.
    bounce_src: Handle<ShaderBuffer>,
    display: Handle<Image>,
}

#[derive(Resource, Deref, DerefMut)]
struct ErrorLogTimer(Timer);

/// GPU timing span names; also the `render/<name>/elapsed_gpu` diagnostics.
fn cascade_span(cascade: u32) -> String {
    format!("rc2d_cascade_{cascade}")
}
fn bounce_pass_span(iteration: u32) -> String {
    format!("rc2d_bounce_pass_{iteration}")
}
const REFERENCE_SPAN: &str = "rc2d_reference";
const RESOLVE_SPAN: &str = "rc2d_resolve";

impl Plugin for Rc2dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Rc2dSettings>()
            .insert_resource(ErrorLogTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
            .add_plugins((
                RenderDiagnosticsPlugin,
                ExtractResourcePlugin::<Rc2dSettings>::default(),
                ExtractResourcePlugin::<Rc2dBuffers>::default(),
            ))
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (input, advance_reference, request_error_readback, log_gpu_timings).chain(),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_pipelines)
            // Before the main pass, so the sprite shows this frame's result.
            // Runs once per camera; this app has exactly one.
            .add_systems(Core2d, run_rc2d.in_set(Core2dSystems::Prepass));
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let mut display = Image::new_fill(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    display.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    display.sampler = ImageSampler::nearest();
    let display = images.add(display);

    let mut buffer = |bytes: u64| {
        buffers.add(ShaderBuffer::with_size(
            bytes as usize,
            RenderAssetUsages::RENDER_WORLD,
        ))
    };
    commands.insert_resource(Rc2dBuffers {
        cascade_a: buffer(CASCADE_ENTRIES * CASCADE_ENTRY_BYTES),
        cascade_b: buffer(CASCADE_ENTRIES * CASCADE_ENTRY_BYTES),
        reference: buffer(PIXELS * VEC4),
        error: buffer(PIXELS * ERROR_BYTES_PER_PIXEL),
        rc_history: buffer(PIXELS * VEC4),
        bounce_src: buffer(CASCADE_ENTRIES * CASCADE_ENTRY_BYTES),
        display: display.clone(),
    });

    commands.insert_resource(ShaderLibraries(
        SHADER_LIBRARIES.iter().map(|path| asset_server.load(*path)).collect(),
    ));

    commands.spawn((Camera2d, Tonemapping::None));
    commands.spawn(Sprite::from_image(display));

    info!("rc2d: 1/2/3 view, B bilinear fix, Up/Down interval, R reset reference, drag light");
}

fn input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor_moved: MessageReader<CursorMoved>,
    mut cursor: Local<Option<Vec2>>,
    mut settings: ResMut<Rc2dSettings>,
) {
    // Window coordinates are y-down with a 1:1 scale factor, same as the
    // canvas, so the cursor position is directly a scene position.
    if let Some(moved) = cursor_moved.read().last() {
        *cursor = Some(moved.position);
    }
    if mouse.pressed(MouseButton::Left)
        && let Some(pos) = *cursor
        && pos != settings.light_pos
    {
        settings.light_pos = pos;
        settings.reset_reference = true;
    }

    for (key, mode) in [(KeyCode::Digit1, 0), (KeyCode::Digit2, 1), (KeyCode::Digit3, 2)] {
        if keys.just_pressed(key) {
            settings.view_mode = mode;
        }
    }
    if keys.just_pressed(KeyCode::KeyB) {
        settings.bilinear_fix = !settings.bilinear_fix;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        settings.base_interval = (settings.base_interval * 2.0).min(16.0);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        settings.base_interval = (settings.base_interval * 0.5).max(1.0 / 16.0);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        settings.reset_reference = true;
    }
    if keys.just_pressed(KeyCode::Space) {
        settings.reference_paused = !settings.reference_paused;
        info!(
            "reference {}",
            if settings.reference_paused { "paused" } else { "resumed" }
        );
    }
    if keys.just_pressed(KeyCode::KeyP) {
        settings.probe0_log2 ^= 1;
    }
    if keys.just_pressed(KeyCode::KeyJ) {
        settings.jitter = !settings.jitter;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        settings.scene ^= 1;
        settings.light_pos = scene_light_start(settings.scene);
        settings.reset_reference = true;
    }
    if keys.just_pressed(KeyCode::KeyG) {
        settings.bounces = (settings.bounces + 1) % 5;
        // The reference path-traces the same number of bounces.
        settings.reset_reference = true;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        settings.coarse_bounce_passes = !settings.coarse_bounce_passes;
    }
    if keys.just_pressed(KeyCode::KeyD) {
        settings.per_direction_levels = match settings.per_direction_levels {
            0..3 => settings.per_direction_levels + 1,
            3 => MAX_CASCADES,
            _ => 0,
        };
    }
    if settings.is_changed() && keys.get_just_pressed().next().is_some() {
        info!(
            "scene={} bounces={} coarse_bounce_passes={} view={} bilinear_fix={} per_direction_levels={} jitter={} s0={} b={} cascades={}",
            settings.scene,
            settings.effective_bounces(),
            settings.coarse_bounce_passes,
            ["cascades", "reference", "difference"][settings.view_mode as usize],
            settings.bilinear_fix,
            settings.per_direction_levels,
            settings.jitter,
            1 << settings.probe0_log2,
            settings.base_interval,
            settings.cascade_count()
        );
    }
}

/// Runs after `input`, so a reset requested this frame is what gets
/// extracted: `ref_frame == 0` tells the shader to overwrite. (Resetting
/// `ref_frame` in `input` directly got incremented away before extraction.)
fn advance_reference(
    mut settings: ResMut<Rc2dSettings>,
    mut last_cascade_key: Local<Option<CascadeKey>>,
) {
    if settings.reset_reference {
        settings.ref_frame = 0;
        settings.reset_reference = false;
        if settings.reference_paused {
            settings.reference_paused = false;
            info!("reference resumed (restarted)");
        }
    } else if !settings.reference_paused {
        settings.ref_frame += 1;
    }

    let key = settings.cascade_key();
    if last_cascade_key.replace(key) != Some(key) {
        settings.rc_frame = 0;
    } else {
        settings.rc_frame += 1;
    }
}

fn log_gpu_timings(
    time: Res<Time>,
    mut timer: Local<Option<Timer>>,
    settings: Res<Rc2dSettings>,
    store: Res<DiagnosticsStore>,
) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }
    let gpu_ms = |span: &str| {
        store
            .get(&DiagnosticPath::new(format!("render/{span}/elapsed_gpu")))
            .and_then(|d| d.smoothed())
    };

    let mut stages: Vec<(String, f64)> = Vec::new();
    let reference_span = (!settings.reference_paused).then(|| REFERENCE_SPAN.to_string());
    for span in reference_span
        .into_iter()
        .chain((0..settings.cascade_count()).rev().map(cascade_span))
        .chain([RESOLVE_SPAN.to_string()])
    {
        let Some(ms) = gpu_ms(&span) else {
            info!("GPU timings unavailable (no timestamp queries inside passes on this adapter?)");
            return;
        };
        stages.push((span.trim_start_matches("rc2d_").to_string(), ms));
    }

    // Earlier bounce passes are timed as a whole; the per-cascade spans
    // above are the final pass.
    for iteration in 0..settings.effective_bounces() {
        if let Some(ms) = gpu_ms(&bounce_pass_span(iteration)) {
            stages.push((format!("bounce_pass_{iteration}"), ms));
        }
    }

    let cascades: f64 = stages
        .iter()
        .filter(|(name, _)| name.starts_with("cascade") || name.starts_with("bounce_pass"))
        .map(|(_, ms)| ms)
        .sum();
    let breakdown: Vec<String> = stages
        .iter()
        .map(|(name, ms)| format!("{name} {ms:.2}"))
        .collect();
    info!(
        "GPU ms | cascades total {cascades:.2} over {} pass(es) (bilinear_fix={}) | {}",
        settings.effective_bounces() + 1,
        settings.bilinear_fix,
        breakdown.join(", ")
    );
}

fn request_error_readback(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<ErrorLogTimer>,
    buffers: Res<Rc2dBuffers>,
) {
    if !timer.tick(time.delta()).just_finished() {
        return;
    }
    commands
        .spawn(Readback::buffer(buffers.error.clone()))
        .observe(
            |event: On<ReadbackComplete>,
             settings: Res<Rc2dSettings>,
             mut commands: Commands,
             mut handled: Local<bool>| {
                // Readback repeats every frame while the entity lives, so a
                // second result can already be in flight when this runs.
                // Each request gets its own observer, so `handled` is per
                // request.
                if std::mem::replace(&mut *handled, true) {
                    return;
                }
                commands.entity(event.entity).try_despawn();

                let (mut err, mut reference, mut pixels, mut rays) = (0.0f64, 0.0f64, 0u32, 0.0f32);
                let mut room_stats = room::RoomStats::default();
                for (i, texel) in event.data.chunks_exact(32).enumerate() {
                    let f = |k: usize| f32::from_le_bytes(texel[k * 4..k * 4 + 4].try_into().unwrap());
                    rays = rays.max(f(3).abs());
                    let (squared_error, squared_reference, reference_sum, cascades_sum) = (f(0), f(1), f(4), f(5));
                    if f(6) > 0.0 {
                        room_stats.add_wall(squared_error, squared_reference, reference_sum, cascades_sum);
                    }
                    if f(3) <= 0.0 {
                        continue;
                    }
                    err += squared_error as f64;
                    reference += squared_reference as f64;
                    pixels += 1;
                    let pos = Vec2::new((i as u32 % SIZE) as f32 + 0.5, (i as u32 / SIZE) as f32 + 0.5);
                    room_stats.add_interior(pos, squared_error, squared_reference, reference_sum, cascades_sum);
                }
                // Before the pipelines compile, the buffer is still zeros;
                // don't report that as a perfect 0% match.
                if pixels == 0 {
                    info!("rc2d: no data yet (pipelines still compiling?)");
                    return;
                }
                let room_stats = if settings.scene == room::SCENE_ID {
                    room_stats.summary()
                } else {
                    String::new()
                };
                info!(
                    "relative RMSE {:.3}% over {pixels} px{room_stats} | scene={} bounces={} coarse_bounce_passes={} bilinear_fix={} per_direction_levels={} jitter={} s0={} b={} cascades={} | reference {rays} rays/px",
                    100.0 * (err / reference.max(f64::MIN_POSITIVE)).sqrt(),
                    settings.scene,
                    settings.effective_bounces(),
                    settings.coarse_bounce_passes,
                    settings.bilinear_fix,
                    settings.per_direction_levels,
                    settings.jitter,
                    1 << settings.probe0_log2,
                    settings.base_interval,
                    settings.cascade_count(),
                );
            },
        );
}

// ---------------------------------------------------------------- render

#[derive(ShaderType, Clone, Copy)]
struct Params {
    size: UVec2,
    cascade: u32,
    cascade_count: u32,
    base_interval: f32,
    merge_mode: u32,
    view_mode: u32,
    ref_frame: u32,
    ref_rays: u32,
    exposure: f32,
    diff_gain: f32,
    light_pos: Vec2,
    probe0_log2: u32,
    ray_rotation: f32,
    rc_accumulate: u32,
    rc_frame: u32,
    per_direction_levels: u32,
    scene_id: u32,
    bounces: u32,
    bounce_iteration: u32,
    bounce_probe0_log2: u32,
}

#[derive(Resource)]
struct Rc2dPipelines {
    layout: BindGroupLayoutDescriptor,
    cascade: CachedComputePipelineId,
    reference: CachedComputePipelineId,
    resolve: CachedComputePipelineId,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let cascade_size = std::num::NonZero::new(CASCADE_ENTRIES * CASCADE_ENTRY_BYTES);
    let pixel_size = std::num::NonZero::new(PIXELS * VEC4);
    let layout = BindGroupLayoutDescriptor::new(
        "rc2d_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Params>(true),
                storage_buffer_read_only_sized(false, cascade_size),
                storage_buffer_sized(false, cascade_size),
                storage_buffer_sized(false, pixel_size),
                storage_buffer_sized(false, std::num::NonZero::new(PIXELS * ERROR_BYTES_PER_PIXEL)),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                storage_buffer_sized(false, pixel_size),
                storage_buffer_read_only_sized(false, cascade_size),
            ),
        ),
    );

    let shader: Handle<Shader> = asset_server.load(SHADER_PATH);
    let queue = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("rc2d_{entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let (cascade, reference, resolve) =
        (queue("cascade"), queue("reference_accumulate"), queue("resolve"));

    commands.insert_resource(Rc2dPipelines {
        layout,
        cascade,
        reference,
        resolve,
    });
}

#[allow(clippy::too_many_arguments)]
fn run_rc2d(
    settings: Res<Rc2dSettings>,
    buffers: Res<Rc2dBuffers>,
    pipelines: Res<Rc2dPipelines>,
    pipeline_cache: Res<PipelineCache>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: Local<DynamicUniformBuffer<Params>>,
    mut ctx: RenderContext,
) {
    let (Some(cascade_pipeline), Some(reference_pipeline), Some(resolve_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.cascade),
        pipeline_cache.get_compute_pipeline(pipelines.reference),
        pipeline_cache.get_compute_pipeline(pipelines.resolve),
    ) else {
        return;
    };
    let (Some(a), Some(b), Some(reference), Some(error), Some(rc_history), Some(bounce_src), Some(display)) = (
        gpu_buffers.get(&buffers.cascade_a),
        gpu_buffers.get(&buffers.cascade_b),
        gpu_buffers.get(&buffers.reference),
        gpu_buffers.get(&buffers.error),
        gpu_buffers.get(&buffers.rc_history),
        gpu_buffers.get(&buffers.bounce_src),
        gpu_images.get(&buffers.display),
    ) else {
        return;
    };

    let cascade_count = settings.cascade_count();
    let base = Params {
        size: UVec2::splat(SIZE),
        cascade: 0,
        cascade_count,
        base_interval: settings.base_interval,
        merge_mode: settings.bilinear_fix as u32,
        view_mode: settings.view_mode,
        ref_frame: settings.ref_frame,
        ref_rays: settings.ref_rays,
        exposure: settings.exposure,
        diff_gain: settings.diff_gain,
        light_pos: settings.light_pos,
        probe0_log2: settings.probe0_log2,
        ray_rotation: settings.ray_rotation(),
        rc_accumulate: settings.jitter as u32,
        rc_frame: settings.rc_frame,
        per_direction_levels: settings.per_direction_levels,
        scene_id: settings.scene,
        bounces: settings.effective_bounces(),
        bounce_iteration: 0,
        bounce_probe0_log2: settings.probe0_log2,
    };
    params.clear();
    // Offsets [iteration][cascade] for the cascade passes; the last one is
    // shared by the reference and resolve passes, which ignore both. Each
    // pass has its own probe spacing and b (see pass_layout), and reads
    // bounce_src at the previous pass's spacing.
    let passes = settings.effective_bounces() + 1;
    let pass_params: Vec<Params> = (0..passes)
        .map(|bounce_iteration| {
            let (probe0_log2, base_interval) = settings.pass_layout(bounce_iteration);
            Params {
                bounce_iteration,
                probe0_log2,
                base_interval,
                cascade_count: cascade_count_for(base_interval),
                bounce_probe0_log2: settings.pass_layout(bounce_iteration.saturating_sub(1)).0,
                ..base
            }
        })
        .collect();
    let cascade_offsets: Vec<Vec<u32>> = pass_params
        .iter()
        .map(|pass| {
            (0..pass.cascade_count)
                .map(|cascade| params.push(&Params { cascade, ..*pass }))
                .collect()
        })
        .collect();
    let shared_offset = params.push(&base);
    params.write_buffer(&render_device, &render_queue);
    let Some(params_binding) = params.binding() else {
        return;
    };

    // Cascades run top-down, ping-ponging: cascade n reads n+1 from `src`
    // and writes `dst`. Even cascades write A, so cascade 0 ends up in A.
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    let bind_group = |src: &Buffer, dst: &Buffer| {
        render_device.create_bind_group(
            "rc2d",
            &layout,
            &BindGroupEntries::sequential((
                params_binding.clone(),
                src.as_entire_buffer_binding(),
                dst.as_entire_buffer_binding(),
                reference.buffer.as_entire_buffer_binding(),
                error.buffer.as_entire_buffer_binding(),
                &display.texture_view,
                rc_history.buffer.as_entire_buffer_binding(),
                bounce_src.buffer.as_entire_buffer_binding(),
            )),
        )
    };
    let writes_a = bind_group(&b.buffer, &a.buffer);
    let writes_b = bind_group(&a.buffer, &b.buffer);

    let pixel_groups = SIZE.div_ceil(8);
    let entries_at = |probe0_log2: u32| (CASCADE_ENTRIES >> (2 * probe0_log2)) as u32;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let pass_descriptor = ComputePassDescriptor {
        label: Some("rc2d"),
        timestamp_writes: None,
    };

    if !settings.reference_paused {
        let mut pass = ctx.command_encoder().begin_compute_pass(&pass_descriptor);
        let span = diagnostics.pass_span(&mut pass, REFERENCE_SPAN);
        pass.set_pipeline(reference_pipeline);
        pass.set_bind_group(0, &writes_a, &[shared_offset]);
        pass.dispatch_workgroups(pixel_groups, pixel_groups, 1);
        span.end(&mut pass);
    }

    // One full cascade pass per bounce, plus the direct pass. Between
    // passes, cascade 0 (in A) is copied to bounce_src, where the next pass
    // reads it to light diffuse surfaces. The final pass gets per-cascade
    // timing spans; earlier ones are timed whole.
    for iteration in 0..passes {
        let last = iteration + 1 == passes;
        let pass_params = &pass_params[iteration as usize];
        let entries = entries_at(pass_params.probe0_log2);
        {
            let mut pass = ctx.command_encoder().begin_compute_pass(&pass_descriptor);
            let whole = (!last).then(|| diagnostics.pass_span(&mut pass, bounce_pass_span(iteration)));
            pass.set_pipeline(cascade_pipeline);
            for cascade in (0..pass_params.cascade_count).rev() {
                let span = last.then(|| diagnostics.pass_span(&mut pass, cascade_span(cascade)));
                let group = if cascade % 2 == 0 { &writes_a } else { &writes_b };
                pass.set_bind_group(0, group, &[cascade_offsets[iteration as usize][cascade as usize]]);
                pass.dispatch_workgroups(entries.div_ceil(64), 1, 1);
                if let Some(span) = span {
                    span.end(&mut pass);
                }
            }
            if let Some(whole) = whole {
                whole.end(&mut pass);
            }
        }
        if !last {
            ctx.command_encoder().copy_buffer_to_buffer(
                &a.buffer,
                0,
                &bounce_src.buffer,
                0,
                entries as u64 * CASCADE_ENTRY_BYTES,
            );
        }
    }

    // Reads cascade 0 (in A) as `src`.
    let mut pass = ctx.command_encoder().begin_compute_pass(&pass_descriptor);
    let span = diagnostics.pass_span(&mut pass, RESOLVE_SPAN);
    pass.set_pipeline(resolve_pipeline);
    pass.set_bind_group(0, &writes_b, &[shared_offset]);
    pass.dispatch_workgroups(pixel_groups, pixel_groups, 1);
    span.end(&mut pass);
}
