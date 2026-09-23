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
//! - `R`: restart reference accumulation
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

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{Core2d, Core2dSystems, tonemapping::Tonemapping},
    image::ImageSampler,
    prelude::*,
    render::{
        RenderApp, RenderStartup,
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

const SIZE: u32 = 512;
const _: () = assert!(SIZE.is_power_of_two(), "probe grids assume exact halving");
const SHADER_PATH: &str = "shaders/rc2d.wgsl";
/// Every cascade stores (W/2^n)(H/2^n) probes * 4^(n+1) rays = 4WH entries.
const CASCADE_ENTRIES: u64 = 4 * SIZE as u64 * SIZE as u64;
const PIXELS: u64 = SIZE as u64 * SIZE as u64;
const VEC4: u64 = 16;
const MAX_CASCADES: u32 = 9;

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
    ref_rays: u32,
    exposure: f32,
    diff_gain: f32,
}

impl Default for Rc2dSettings {
    fn default() -> Self {
        Self {
            view_mode: 0,
            bilinear_fix: true,
            // Cascade 0 covers [0, 2px] at 1px probe spacing. Measured best
            // for both merge modes on this scene; see the module docs.
            base_interval: 0.5,
            ref_frame: 0,
            ref_rays: 16,
            exposure: 2.0,
            diff_gain: 8.0,
        }
    }
}

impl Rc2dSettings {
    /// Fewest cascades whose last interval end, b*4^N, reaches the canvas
    /// diagonal, so every ray can see the whole scene.
    fn cascade_count(&self) -> u32 {
        let diagonal = (2.0 * (SIZE as f32).powi(2)).sqrt();
        (1..=MAX_CASCADES)
            .find(|&n| self.base_interval * 4f32.powi(n as i32) >= diagonal)
            .unwrap_or(MAX_CASCADES)
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct Rc2dBuffers {
    cascade_a: Handle<ShaderBuffer>,
    cascade_b: Handle<ShaderBuffer>,
    reference: Handle<ShaderBuffer>,
    error: Handle<ShaderBuffer>,
    display: Handle<Image>,
}

#[derive(Resource, Deref, DerefMut)]
struct ErrorLogTimer(Timer);

impl Plugin for Rc2dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Rc2dSettings>()
            .insert_resource(ErrorLogTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
            .add_plugins((
                ExtractResourcePlugin::<Rc2dSettings>::default(),
                ExtractResourcePlugin::<Rc2dBuffers>::default(),
            ))
            .add_systems(Startup, setup)
            .add_systems(Update, (input, advance_reference, request_error_readback).chain());

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

    let mut buffer = |entries: u64| {
        buffers.add(ShaderBuffer::with_size(
            (entries * VEC4) as usize,
            RenderAssetUsages::RENDER_WORLD,
        ))
    };
    commands.insert_resource(Rc2dBuffers {
        cascade_a: buffer(CASCADE_ENTRIES),
        cascade_b: buffer(CASCADE_ENTRIES),
        reference: buffer(PIXELS),
        error: buffer(PIXELS),
        display: display.clone(),
    });

    commands.spawn((Camera2d, Tonemapping::None));
    commands.spawn(Sprite::from_image(display));

    info!("rc2d: 1/2/3 view, B bilinear fix, Up/Down interval, R reset reference");
}

fn input(keys: Res<ButtonInput<KeyCode>>, mut settings: ResMut<Rc2dSettings>) {
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
        settings.ref_frame = 0;
    }
    if settings.is_changed() && keys.get_just_pressed().next().is_some() {
        info!(
            "view={} bilinear_fix={} b={} cascades={}",
            ["cascades", "reference", "difference"][settings.view_mode as usize],
            settings.bilinear_fix,
            settings.base_interval,
            settings.cascade_count()
        );
    }
}

/// The frame just extracted used `ref_frame`; the next one accumulates on
/// top of it.
fn advance_reference(mut settings: ResMut<Rc2dSettings>) {
    settings.ref_frame += 1;
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
            |event: On<ReadbackComplete>, settings: Res<Rc2dSettings>, mut commands: Commands| {
                // Readback repeats every frame while the entity lives, and a
                // second result can already be in flight when this runs.
                commands.entity(event.entity).try_despawn();

                let (mut err, mut reference, mut pixels, mut rays) = (0.0f64, 0.0f64, 0u32, 0.0f32);
                for texel in event.data.chunks_exact(16) {
                    let f = |i: usize| f32::from_le_bytes(texel[i * 4..i * 4 + 4].try_into().unwrap());
                    err += f(0) as f64;
                    reference += f(1) as f64;
                    pixels += f(2) as u32;
                    rays = rays.max(f(3));
                }
                // Before the pipelines compile, the buffer is still zeros;
                // don't report that as a perfect 0% match.
                if pixels == 0 {
                    info!("rc2d: no data yet (pipelines still compiling?)");
                    return;
                }
                info!(
                    "relative RMSE {:.3}% over {pixels} px | bilinear_fix={} b={} cascades={} | reference {rays} rays/px",
                    100.0 * (err / reference.max(f64::MIN_POSITIVE)).sqrt(),
                    settings.bilinear_fix,
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
    let cascade_size = std::num::NonZero::new(CASCADE_ENTRIES * VEC4);
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
                storage_buffer_sized(false, pixel_size),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
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
    let (Some(a), Some(b), Some(reference), Some(error), Some(display)) = (
        gpu_buffers.get(&buffers.cascade_a),
        gpu_buffers.get(&buffers.cascade_b),
        gpu_buffers.get(&buffers.reference),
        gpu_buffers.get(&buffers.error),
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
    };
    params.clear();
    // Offsets [0..cascade_count) are per-cascade; the last one is shared by
    // the reference and resolve passes, which ignore `cascade`.
    let cascade_offsets: Vec<u32> = (0..cascade_count)
        .map(|cascade| params.push(&Params { cascade, ..base }))
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
            )),
        )
    };
    let writes_a = bind_group(&b.buffer, &a.buffer);
    let writes_b = bind_group(&a.buffer, &b.buffer);

    let pixel_groups = SIZE.div_ceil(8);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("rc2d"),
            timestamp_writes: None,
        });

    pass.set_pipeline(reference_pipeline);
    pass.set_bind_group(0, &writes_a, &[shared_offset]);
    pass.dispatch_workgroups(pixel_groups, pixel_groups, 1);

    pass.set_pipeline(cascade_pipeline);
    for cascade in (0..cascade_count).rev() {
        let group = if cascade % 2 == 0 { &writes_a } else { &writes_b };
        pass.set_bind_group(0, group, &[cascade_offsets[cascade as usize]]);
        pass.dispatch_workgroups((CASCADE_ENTRIES as u32).div_ceil(64), 1, 1);
    }

    // Reads cascade 0 (in A) as `src`.
    pass.set_pipeline(resolve_pipeline);
    pass.set_bind_group(0, &writes_b, &[shared_offset]);
    pass.dispatch_workgroups(pixel_groups, pixel_groups, 1);
}
