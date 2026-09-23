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
//! - `L`: cycle how many of the lowest cascades use the per-direction
//!   merge (0-3, then all; see the cascade shader)
//! - `Tab`: switch scene: lights and occluders / room with thin diffuse walls,
//!   a spotlight outside its only entrance and lights around its outside
//!   (resets the light position)
//! - `G`: cycle diffuse bounces 0-4 (room scene; restarts the reference)
//! - `WASD` / right-drag: move the circle of live probes (eventually the
//!   camera's region); `-` / `=`: shrink / grow it. Probes that stay in the
//!   circle keep their cached hits; only ones entering it are re-marched.
//! - `T`: throw away the hit cache and re-march every probe (for checking
//!   that scrolled results match a fresh trace)
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
//! adapter supports timestamps inside compute passes, and how many probes
//! each cascade re-marched that frame.
//!
//! Tracing and gathering are separate passes: tracing caches each ray
//! segment's hit location, gathering evaluates lighting at the cached hits
//! every frame (once per bounce). Geometry or cascade-layout changes
//! (scene, moving the light, which also occludes, b, s0, merge settings,
//! circle radius) invalidate the cache; lighting-only changes and moving
//! the circle don't.
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
//! shader). Merging per child direction in the lowest cascades (`L`):
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
//! Room scene (`Tab`), 2 diffuse bounces, same-run comparisons at ~12-17k
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
//! intermediate passes at double probe spacing cut cascade work 37.1 ->
//! 24.5 ms with every error figure within the spread of the off runs
//! (in-room 16.2% vs 15.7-16.5%, wall faces -6.3% vs -6.4%). That option
//! was removed once tracing got cached: it existed to make re-tracing per
//! bounce cheaper, and bounce passes are now gathers over one trace.
//! Compiling the bounce lookup out of the direct pass was also
//! tried: ~2-3%, within noise, so not kept. The path-traced reference, at
//! ~53 ms, is the largest cost in the room scene.
//!
//! Hit cache (circle radius 160 unless noted): with the circle still, no
//! probe re-marches (trace ~0.22 ms, just the stale check). Moving at 120
//! px/s re-marches only the leading edge, ~150-160 cascade-0 probes per
//! frame of ~20k in the circle, and far fewer in coarser cascades, which
//! scroll in bigger steps. A full re-march (`T`) after scrolling continued
//! the error's smooth convergence trend with no jump, so scrolled results
//! match a fresh trace. Room scene, circle covering the whole canvas: error
//! figures identical to the uncached runs (7.68%, in-room -1.22%, wall
//! faces -6.39%), but cascade work is 32.7 ms vs 24.5 ms with the removed
//! coarse passes: the direct gather is ~5.9 ms, each bounce gather 10.8-15.8
//! ms. The bounce lookup per diffuse hit (normal from 4 SDF evaluations, an
//! SDF test per probe, 16 loads) now dominates, not marching.
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

mod cache;
mod room;

use cache::{CacheLayout, LayoutInputs};

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
    /// Circle of live probes; see the WASD control.
    circle_center: Vec2,
    circle_radius: f32,
    /// Bumped whenever the hit cache must be thrown away; the render world
    /// clears the tags when it sees a new value.
    cache_generation: u32,
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
            circle_center: Vec2::splat(SIZE as f32 / 2.0),
            circle_radius: 160.0,
            cache_generation: 0,
            rc_frame: 0,
        }
    }
}

/// Everything the cached hits depend on: geometry (scene and the light,
/// which also occludes) and which segments each ray marches.
type HitCacheKey = (u32, bool, u32, u32, [u32; 2], u32, u32);
/// Everything the displayed cascades depend on; restarts the jitter average.
type CascadeKey = (HitCacheKey, bool, u32, [u32; 2]);

impl Rc2dSettings {
    fn hit_cache_key(&self) -> HitCacheKey {
        (
            self.base_interval.to_bits(),
            self.bilinear_fix,
            self.per_direction_levels,
            self.probe0_log2,
            [self.light_pos.x.to_bits(), self.light_pos.y.to_bits()],
            self.scene,
            self.cache_layout().radius.to_bits(),
        )
    }

    fn cache_layout(&self) -> CacheLayout {
        CacheLayout::new(&LayoutInputs {
            canvas: SIZE,
            probe0_log2: self.probe0_log2,
            cascade_count: self.cascade_count(),
            per_direction_levels: self.per_direction_levels,
            bilinear_fix: self.bilinear_fix,
            radius: self.circle_radius,
        })
    }

    fn effective_bounces(&self) -> u32 {
        if self.scene == room::SCENE_ID { self.bounces } else { 0 }
    }

    /// Everything the cascades' output depends on; a change restarts the
    /// jittered average.
    fn cascade_key(&self) -> CascadeKey {
        (
            self.hit_cache_key(),
            self.jitter,
            self.effective_bounces(),
            [self.circle_center.x.to_bits(), self.circle_center.y.to_bits()],
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

#[derive(Resource, Clone, ExtractResource)]
struct Rc2dBuffers {
    cascade_a: Handle<ShaderBuffer>,
    cascade_b: Handle<ShaderBuffer>,
    reference: Handle<ShaderBuffer>,
    error: Handle<ShaderBuffer>,
    rc_history: Handle<ShaderBuffer>,
    /// Cascade 0 of the previous bounce pass.
    bounce_src: Handle<ShaderBuffer>,
    /// Hit cache: cached hit fractions, slot tags, per-cascade re-march
    /// counters.
    hits: Handle<ShaderBuffer>,
    tags: Handle<ShaderBuffer>,
    remarched: Handle<ShaderBuffer>,
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
const TRACE_SPAN: &str = "rc2d_trace";
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
                (
                    input,
                    advance_reference,
                    request_error_readback,
                    log_gpu_timings,
                    request_remarch_readback,
                )
                    .chain(),
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
        hits: buffer(cache::HIT_BUDGET_BYTES),
        tags: buffer(cache::TAG_CAPACITY_WORDS * 4),
        remarched: buffer(MAX_CASCADES as u64 * 4),
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
    time: Res<Time>,
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
    if mouse.pressed(MouseButton::Right)
        && let Some(pos) = *cursor
    {
        settings.circle_center = pos;
    }
    let mut step = Vec2::ZERO;
    for (key, dir) in [
        (KeyCode::KeyW, Vec2::NEG_Y),
        (KeyCode::KeyS, Vec2::Y),
        (KeyCode::KeyA, Vec2::NEG_X),
        (KeyCode::KeyD, Vec2::X),
    ] {
        if keys.pressed(key) {
            step += dir;
        }
    }
    if step != Vec2::ZERO {
        let speed = 120.0;
        settings.circle_center = (settings.circle_center + step.normalize() * speed * time.delta_secs())
            .clamp(Vec2::ZERO, Vec2::splat(SIZE as f32));
    }
    if keys.just_pressed(KeyCode::Minus) {
        settings.circle_radius = (settings.circle_radius - 16.0).max(32.0);
    }
    if keys.just_pressed(KeyCode::Equal) {
        settings.circle_radius = (settings.circle_radius + 16.0).min(368.0);
    }
    if keys.just_pressed(KeyCode::KeyT) {
        settings.cache_generation += 1;
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
    if keys.just_pressed(KeyCode::Tab) {
        settings.scene ^= 1;
        settings.light_pos = scene_light_start(settings.scene);
        settings.reset_reference = true;
    }
    if keys.just_pressed(KeyCode::KeyG) {
        settings.bounces = (settings.bounces + 1) % 5;
        // The reference path-traces the same number of bounces.
        settings.reset_reference = true;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        settings.per_direction_levels = match settings.per_direction_levels {
            0..3 => settings.per_direction_levels + 1,
            3 => MAX_CASCADES,
            _ => 0,
        };
    }
    if settings.is_changed() && keys.get_just_pressed().next().is_some() {
        info!(
            "scene={} bounces={} circle_radius={} view={} bilinear_fix={} per_direction_levels={} jitter={} s0={} b={} cascades={}",
            settings.scene,
            settings.effective_bounces(),
            settings.cache_layout().radius,
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
    mut last_hit_key: Local<Option<HitCacheKey>>,
    mut warned_radius: Local<f32>,
) {
    // Jitter rotates every ray each frame, so nothing cached stays valid.
    let hit_key = settings.hit_cache_key();
    if last_hit_key.replace(hit_key) != Some(hit_key) || settings.jitter {
        settings.cache_generation += 1;
    }
    let radius = settings.cache_layout().radius;
    if radius < settings.circle_radius && radius != *warned_radius {
        warn!(
            "circle radius {} doesn't fit the {} MiB hit cache with these settings; using {radius}",
            settings.circle_radius,
            cache::HIT_BUDGET_BYTES >> 20
        );
        *warned_radius = radius;
    }

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
        .chain([TRACE_SPAN.to_string()])
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
        .filter(|(name, _)| {
            name.starts_with("cascade") || name.starts_with("bounce_pass") || name == "trace"
        })
        .map(|(_, ms)| ms)
        .sum();
    let breakdown: Vec<String> = stages
        .iter()
        .map(|(name, ms)| format!("{name} {ms:.2}"))
        .collect();
    info!(
        "GPU ms | cascades total {cascades:.2} (trace + {} gather pass(es), bilinear_fix={}) | {}",
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
                    "relative RMSE {:.3}% over {pixels} px in the circle{room_stats} | scene={} bounces={} bilinear_fix={} per_direction_levels={} jitter={} s0={} b={} cascades={} | reference {rays} rays/px",
                    100.0 * (err / reference.max(f64::MIN_POSITIVE)).sqrt(),
                    settings.scene,
                    settings.effective_bounces(),
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

/// Logs how many probes each cascade re-marched in one frame: zero while
/// the circle is still, only the entering edge while it moves.
fn request_remarch_readback(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: Local<Option<Timer>>,
    buffers: Res<Rc2dBuffers>,
) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }
    commands.spawn(Readback::buffer(buffers.remarched.clone())).observe(
        |event: On<ReadbackComplete>,
         settings: Res<Rc2dSettings>,
         mut commands: Commands,
         mut handled: Local<bool>| {
            if std::mem::replace(&mut *handled, true) {
                return;
            }
            commands.entity(event.entity).try_despawn();
            let layout = settings.cache_layout();
            let counts: Vec<String> = event
                .data
                .chunks_exact(4)
                .zip(&layout.cascades)
                .enumerate()
                .map(|(n, (word, window))| {
                    let count = u32::from_le_bytes(word.try_into().unwrap());
                    format!("c{n} {count}/{}", window.size * window.size)
                })
                .collect();
            info!("re-marched probes this frame (of window slots): {}", counts.join(", "));
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
    circle_center: Vec2,
    window: u32,
    radius: f32,
    tag_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    c0_window: u32,
    c0_radius: f32,
    c0_tag_base: u32,
    hit_base: u32,
}

#[derive(Resource)]
struct Rc2dPipelines {
    layout: BindGroupLayoutDescriptor,
    trace: CachedComputePipelineId,
    commit: CachedComputePipelineId,
    gather: CachedComputePipelineId,
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
                storage_buffer_sized(false, std::num::NonZero::new(cache::HIT_BUDGET_BYTES)),
                storage_buffer_sized(false, std::num::NonZero::new(cache::TAG_CAPACITY_WORDS * 4)),
                storage_buffer_sized(false, std::num::NonZero::new(MAX_CASCADES as u64 * 4)),
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

    commands.insert_resource(Rc2dPipelines {
        layout: layout.clone(),
        trace: queue("cascade_trace"),
        commit: queue("cascade_commit"),
        gather: queue("cascade_gather"),
        reference: queue("reference_accumulate"),
        resolve: queue("resolve"),
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
    mut cleared_generation: Local<Option<u32>>,
    mut ctx: RenderContext,
) {
    let (Some(trace_pipeline), Some(commit_pipeline), Some(gather_pipeline), Some(reference_pipeline), Some(resolve_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.trace),
        pipeline_cache.get_compute_pipeline(pipelines.commit),
        pipeline_cache.get_compute_pipeline(pipelines.gather),
        pipeline_cache.get_compute_pipeline(pipelines.reference),
        pipeline_cache.get_compute_pipeline(pipelines.resolve),
    ) else {
        return;
    };
    let gpu = |handle: &Handle<ShaderBuffer>| gpu_buffers.get(handle).map(|b| b.buffer.clone());
    let (Some(a), Some(b), Some(reference), Some(error), Some(rc_history), Some(bounce_src), Some(hits), Some(tags), Some(remarched), Some(display)) = (
        gpu(&buffers.cascade_a),
        gpu(&buffers.cascade_b),
        gpu(&buffers.reference),
        gpu(&buffers.error),
        gpu(&buffers.rc_history),
        gpu(&buffers.bounce_src),
        gpu(&buffers.hits),
        gpu(&buffers.tags),
        gpu(&buffers.remarched),
        gpu_images.get(&buffers.display),
    ) else {
        return;
    };

    let cascade_count = settings.cascade_count();
    let layout = settings.cache_layout();
    let windows = &layout.cascades;
    let c0 = windows[0];
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
        circle_center: settings.circle_center,
        window: c0.size,
        radius: c0.radius,
        tag_base: c0.tag_base,
        child_window: 0,
        child_radius: 0.0,
        child_tag_base: 0,
        c0_window: c0.size,
        c0_radius: c0.radius,
        c0_tag_base: c0.tag_base,
        hit_base: c0.hit_base,
    };
    params.clear();
    // Offsets [iteration][cascade]; iteration 0's also drive trace and
    // commit. The last offset is shared by the reference and resolve passes.
    let passes = settings.effective_bounces() + 1;
    let cascade_params = |cascade: u32, bounce_iteration: u32| {
        let own = windows[cascade as usize];
        let child = windows.get(cascade as usize + 1).copied().unwrap_or_default();
        Params {
            cascade,
            bounce_iteration,
            window: own.size,
            radius: own.radius,
            tag_base: own.tag_base,
            hit_base: own.hit_base,
            child_window: child.size,
            child_radius: child.radius,
            child_tag_base: child.tag_base,
            ..base
        }
    };
    let offsets: Vec<Vec<u32>> = (0..passes)
        .map(|iteration| {
            (0..cascade_count)
                .map(|cascade| params.push(&cascade_params(cascade, iteration)))
                .collect()
        })
        .collect();
    let shared_offset = params.push(&base);
    params.write_buffer(&render_device, &render_queue);
    let Some(params_binding) = params.binding() else {
        return;
    };

    // Gathers run top-down, ping-ponging: cascade n reads n+1 from `src`
    // and writes `dst`. Even cascades write A, so cascade 0 ends up in A.
    let bind_group_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    let bind_group = |src: &Buffer, dst: &Buffer| {
        render_device.create_bind_group(
            "rc2d",
            &bind_group_layout,
            &BindGroupEntries::sequential((
                params_binding.clone(),
                src.as_entire_buffer_binding(),
                dst.as_entire_buffer_binding(),
                reference.as_entire_buffer_binding(),
                error.as_entire_buffer_binding(),
                &display.texture_view,
                rc_history.as_entire_buffer_binding(),
                bounce_src.as_entire_buffer_binding(),
                hits.as_entire_buffer_binding(),
                tags.as_entire_buffer_binding(),
                remarched.as_entire_buffer_binding(),
            )),
        )
    };
    let writes_a = bind_group(&b, &a);
    let writes_b = bind_group(&a, &b);

    // Invalidation: zeroed tags match no probe, so everything re-marches.
    if *cleared_generation != Some(settings.cache_generation) {
        ctx.command_encoder().clear_buffer(&tags, 0, None);
        *cleared_generation = Some(settings.cache_generation);
    }
    ctx.command_encoder().clear_buffer(&remarched, 0, None);

    let pixel_groups = SIZE.div_ceil(8);
    let slots = |cascade: u32| windows[cascade as usize].size.pow(2);
    let rays = |cascade: u32| 4u32 << (2 * cascade);
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

    // Trace: re-march stale probes of every cascade, then commit the tags.
    {
        let mut pass = ctx.command_encoder().begin_compute_pass(&pass_descriptor);
        let span = diagnostics.pass_span(&mut pass, TRACE_SPAN);
        pass.set_pipeline(trace_pipeline);
        for cascade in 0..cascade_count {
            pass.set_bind_group(0, &writes_a, &[offsets[0][cascade as usize]]);
            pass.dispatch_workgroups((slots(cascade) * rays(cascade)).div_ceil(64), 1, 1);
        }
        pass.set_pipeline(commit_pipeline);
        for cascade in 0..cascade_count {
            pass.set_bind_group(0, &writes_a, &[offsets[0][cascade as usize]]);
            pass.dispatch_workgroups(slots(cascade).div_ceil(64), 1, 1);
        }
        span.end(&mut pass);
    }

    // One gather per bounce, plus the direct one, all over the same cached
    // hits. Between passes, cascade 0 (in A) is copied to bounce_src, where
    // the next pass reads it to light diffuse surfaces. The final pass gets
    // per-cascade timing spans; earlier ones are timed whole.
    for iteration in 0..passes {
        let last = iteration + 1 == passes;
        {
            let mut pass = ctx.command_encoder().begin_compute_pass(&pass_descriptor);
            let whole = (!last).then(|| diagnostics.pass_span(&mut pass, bounce_pass_span(iteration)));
            pass.set_pipeline(gather_pipeline);
            for cascade in (0..cascade_count).rev() {
                let span = last.then(|| diagnostics.pass_span(&mut pass, cascade_span(cascade)));
                let group = if cascade % 2 == 0 { &writes_a } else { &writes_b };
                pass.set_bind_group(0, group, &[offsets[iteration as usize][cascade as usize]]);
                pass.dispatch_workgroups((slots(cascade) * rays(cascade)).div_ceil(64), 1, 1);
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
                &a,
                0,
                &bounce_src,
                0,
                (slots(0) * rays(0)) as u64 * CASCADE_ENTRY_BYTES,
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
