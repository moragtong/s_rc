//! 2.5D radiance cascades over the dual heightmap.
//!
//! The two halves of the work have very different clocks. Where each probe ray
//! lands depends only on the scene, so it is marched **once** into
//! `hits_buffer` by `shaders/march_hits.wgsl`, after the bake and the winding
//! map are final. What arrives *at* those hits depends on the sun, so the
//! light, trace and gather passes relight every [`RELIGHT_INTERVAL`] frames.
//! Every frame, `shaders/blend_sh.wgsl` mixes the previous relight into the
//! current one over the whole probe set, and `shade` reads that single buffer —
//! the cross-fade it would otherwise do per SH coefficient costs the shade pass
//! a second storage read 1.4 million times over. The two relights live on three
//! rotating probe buffers, so a relight gathers into the one the blend will
//! read at weight 0 next and the previous relight survives the swap: the role
//! rotation replaces the copy that used to keep it.

pub mod layout;

use std::sync::atomic::{AtomicU32, Ordering};

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_2d, texture_storage_2d,
                uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use crate::heightmap::Heightmap;
use crate::heightmap_bake::BakeState;
use crate::ui::{BlendToggleState, LightingState};

/// Global variable controlling the number of times the spatial low-pass blur is applied.
pub static LOWPASS_BLUR_PASSES: AtomicU32 = AtomicU32::new(7);
pub const MAX_BLUR_PASSES: usize = 64;

/// Global variable controlling the cross-layer blur radius in tenths of a meter (40 = 4.0m).
pub static CROSS_LAYER_RADIUS_TENTHS: AtomicU32 = AtomicU32::new(20);
pub const MAX_CROSS_LAYER_TENTHS: u32 = 80;

/// Global variable controlling the number of diffuse GI bounces (1 to 4).
pub static BOUNCE_COUNT: AtomicU32 = AtomicU32::new(1);
pub const MAX_BOUNCE_COUNT: u32 = 4;

/// Global variable controlling the number of temporal blur passes / relight interval (1 to 16 frames).
pub static TEMPORAL_BLUR_PASSES: AtomicU32 = AtomicU32::new(16);
pub const MAX_TEMPORAL_PASSES: u32 = 16;

/// Global variable controlling the number of temporal bounce frames for EMA accumulation (1 to 16 frames).
pub static TEMPORAL_BOUNCE_PASSES: AtomicU32 = AtomicU32::new(4);
pub const MAX_TEMPORAL_BOUNCE_PASSES: u32 = 16;

#[derive(ShaderType, Clone, Copy, Default)]
struct LightUniforms {
    pub bounce_index: u32,
    pub temporal_bounce: u32,
    pub temporal_weight: f32,
    pub pad0: u32,
}

const TRACE_LIBRARY_PATH: &str = "shaders/heightmap_trace.wgsl";
const LIGHT_SHADER_PATH: &str = "shaders/surface_light.wgsl";
const TRACE_SHADER_PATH: &str = "shaders/cascades2_5d.wgsl";
const GATHER_SHADER_PATH: &str = "shaders/gather_cascades.wgsl";
const MARCH_SHADER_PATH: &str = "shaders/march_hits.wgsl";
const BLEND_SHADER_PATH: &str = "shaders/blend_sh.wgsl";
const PROBE_POSITIONS_SHADER_PATH: &str = "shaders/probe_positions.wgsl";
const BAKE_PROBE_VIS_SHADER_PATH: &str = "shaders/bake_probe_vis.wgsl";
const BAKE_PROBE_ADJ_SHADER_PATH: &str = "shaders/bake_probe_adj.wgsl";
/// Workgroup sizes: 8x8 texels for the light pass, 64 rays or probes for
/// the others.
const LIGHT_WORKGROUP: u32 = 8;
const WORKGROUP: u32 = 64;
const BLEND_WORKGROUP: u32 = 128;
/// SH probe stride in vec4s: 9 coefficients, then (position, radius).
pub const SH_STRIDE: u64 = 5;
/// Slots per ray in the one-shot hit cache: the top cascade's own trace, or
/// the four segment traces the cascades below it use instead.
const HITS_PER_RAY: u64 = 4;
/// Frames between two relights. The sun moves every frame, so this is the
/// period the indirect light lags behind the direct light by.
pub const RELIGHT_INTERVAL: u32 = 30;
/// Which of the three rotating probe buffers plays each role, per
/// [`Relight::phase`]. `A` is `blend_sh`'s `cur` operand (weight `alpha`, the
/// freshest relight), `B` its `prev` operand (weight `1 - alpha`), and `G`
/// the buffer the next gather overwrites. A relight therefore lands on the
/// gather target, becomes `A` at weight 0 on that same frame, climbs with
/// `alpha` through the interval, and hits weight 1 "by itself" on the next
/// sync, when the swap turns it into `B`.
const A_OF_PHASE: [usize; 3] = [0, 2, 1];
#[allow(dead_code)]
const B_OF_PHASE: [usize; 3] = [1, 0, 2];
const G_OF_PHASE: [usize; 3] = [2, 1, 0];

pub struct Cascades2_5dPlugin;

/// Where the frame counter sits in [`RELIGHT_INTERVAL`], how far through the
/// cross-fade the probes are (0.0 on the frame they were rewritten, climbing
/// to just under 1.0 before the next relight), and which step of the
/// three-way probe-buffer rotation the relights are on. `alpha` is written
/// into `blend_sh`'s uniform, which is what makes the cross-fade a pre-pass
/// over the probe set rather than a per-coefficient read in the fragment
/// shader.
#[derive(Resource, Clone, Copy)]
pub struct Relight {
    pub interval: u32,
    pub alpha: f32,
    /// Rotated on every relight, selecting which of the three probe buffers
    /// the next gather overwrites and which two `blend_sh` reads this frame.
    pub phase: u32,
}

impl Default for Relight {
    fn default() -> Self {
        let interval = std::env::var("S_RC_RELIGHT_INTERVAL")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(RELIGHT_INTERVAL);
        Self {
            interval,
            alpha: 0.0,
            phase: 0,
        }
    }
}

/// What `blend_sh` mixes with. Sized to a whole 16 bytes on purpose: the
/// two scalars would be 8 bytes naturally, but uniform address space rounds a
/// struct up to 16, and the two sides have to agree on the binding's minimum
/// size or the bind group fails validation.
#[derive(ShaderType, Clone, Copy, Default)]
struct BlendUniforms {
    pub alpha: f32,
    pub probe_words: u32,
    pub lowpass: u32,
    pub pass_index: u32,
    pub cross_layer_radius: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct CascadesLayout {
    pub layout: layout::Layout,
}
impl Default for CascadesLayout {
    fn default() -> Self {
        Self {
            layout: layout::Layout::new(),
        }
    }
}

/// The scene's directional light, in the units the cascades work in.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SunParams {
    /// Unit vector toward the sun.
    pub toward: Vec3,
    /// Irradiance times the camera exposure, so cascade radiance matches the
    /// main pass's output.
    pub irradiance: Vec3,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub struct CascadeInfoUniform {
    pub size: UVec2,
    pub interval: Vec2,
    pub ray_base: u32,
    pub probe_base: u32,
    pub spacing: f32,
    pub octa_side: u32,
    pub sub: u32,
    pub sub_base: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub struct CascadesUniforms {
    pub hmap_min: Vec2,
    pub hmap_max: Vec2,
    pub hmap_res: UVec2,
    pub cascade_count: u32,
    pub hmap_top: f32,
    pub split: f32,
    pub sun_toward: Vec3,
    pub sun_irradiance: Vec3,
    pub sky_color: Vec4,
    pub cascades: [CascadeInfoUniform; layout::CASCADES as usize],
}

#[derive(Resource)]
struct CascadesPipeline {
    /// Keeps the shared WGSL module loaded so the passes can import it.
    _trace_library: Handle<Shader>,
    light_layout: BindGroupLayoutDescriptor,
    trace_layout: BindGroupLayoutDescriptor,
    gather_layout: BindGroupLayoutDescriptor,
    march_layout: BindGroupLayoutDescriptor,
    probe_positions_layout: BindGroupLayoutDescriptor,
    bake_probe_adj_layout: BindGroupLayoutDescriptor,
    blend_layout: BindGroupLayoutDescriptor,
    light_pipeline: CachedComputePipelineId,
    trace_pipelines: Vec<CachedComputePipelineId>,
    gather_pipelines: Vec<CachedComputePipelineId>,
    march_pipelines: Vec<CachedComputePipelineId>,
    probe_positions_pipelines: Vec<CachedComputePipelineId>,
    bake_probe_vis_pipeline: CachedComputePipelineId,
    bake_probe_adj_pipelines: Vec<CachedComputePipelineId>,
    blend_pipeline: CachedComputePipelineId,
    bake_probe_vis_layout: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct CascadesRenderData {
    pub rays_buffer: Buffer,
    /// Every probe's final position and baked normal, two vec4s per
    /// probe-with-layer (position pair, normal pair), written once by the
    /// probe_positions pass after the bake and the winding map are final.
    /// Marshalled here so `probe_position`/`probe_normal` in march/trace/
    /// gather/shade do not rerun the disk scan every frame.
    pub probe_positions_buffer: Buffer,
    /// Three rotating SH probe buffers. Which holds the freshest relight, which
    /// the last one and which is the next gather target follows [`Relight::phase`]
    /// through [`A_OF_PHASE`], [`B_OF_PHASE`] and [`G_OF_PHASE`]; at each relight
    /// `G` becomes `A`, `A` becomes `B` and `B` becomes `G`, so the fresh relight
    /// fades in over the interval while the old one fades out - no copy.
    pub sh_probes: [Buffer; 3],
    /// What `shade` reads: the cross-fade between the two relights, written
    /// once per frame over the whole probe set.
    pub sh_probes_blend_buffer: Buffer,
    /// Scratch buffer for ping-pong multi-pass low-pass filtering.
    pub sh_probes_scratch_buffer: Buffer,
    /// Pre-baked 9-bit probe-to-probe visibility adjacency mask.
    pub probe_adj_buffer: Buffer,
    /// Where each ray landed, written once by `march_hits.wgsl`.
    pub hits_buffer: Buffer,
    /// Persistent so a moving sun rewrites its contents instead of allocating
    /// a new buffer every frame.
    pub uniform_buffer: Buffer,
    /// `LightUniforms` buffer per bounce.
    light_uniform_buffers: [Buffer; MAX_BOUNCE_COUNT as usize],
    /// `BlendUniforms`, rewritten every frame per pass.
    blend_uniform_buffers: [Buffer; MAX_BLUR_PASSES],
    subsamples_buffer: Buffer,
    /// Group 0 of every pass: the heightmap and the uniforms.
    pub common_bind_group: BindGroup,
    /// Same slots as `common_bind_group`, but with `probe_positions` bound to
    /// a scratch slab so the pass that writes that buffer can declare the same
    /// import module without binding the destination twice.
    pub probe_positions_pass_bind_group: BindGroup,
    pub probe_positions_bind_group: BindGroup,
    pub probe_vis_pass_bind_group: BindGroup,
    pub probe_vis_bind_group: BindGroup,
    pub probe_adj_bind_group: BindGroup,
    pub sh_probes_history: [Buffer; 2],
    pub bounce_history_buffers: [Buffer; 2],
    /// Light bind group per bounce history index, bounce index and phase.
    light_bind_groups: [[[BindGroup; 3]; MAX_BOUNCE_COUNT as usize]; 2],
    trace_bind_group: BindGroup,
    /// One per rotation phase, writing into the phase's gather target.
    gather_bind_groups: [BindGroup; 3],
    march_bind_group: BindGroup,
    /// Pass 0 Temporal Accumulation: reads history[h] and fresh gather[p], writes history[1 - h].
    blend_temporal_bind_groups: [[BindGroup; 3]; 2],
    /// Pass 1 Spatial Blur reading history[h_curr], writing to scratch.
    blend_spatial_p1_to_scratch: [BindGroup; 2],
    /// Pass 1 Spatial Blur reading history[h_curr], writing to blend.
    blend_spatial_p1_to_blend: [BindGroup; 2],
    /// Successive ping pass: scratch -> blend.
    blend_ping: [BindGroup; MAX_BLUR_PASSES],
    /// Successive pong pass: blend -> scratch.
    blend_pong: [BindGroup; MAX_BLUR_PASSES],
}

/// Group 0, shared with the shading pass.
pub fn common_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "cascades_common_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // heightmap
                uniform_buffer::<CascadesUniforms>(false),
                // Ceiling and floor surface normals. Probe relocation pushes a
                // probe out of the surface it is buried behind, along that
                // surface's normal, so the trace and gather passes need them
                // alongside the heights they came from.
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                // 2D winding number of the split cut: the horizontal
                // containment signal, since the floor/ceiling map cannot
                // record a vertical wall and `is_solid` therefore cannot see
                // one. Written on the CPU by `winding`.
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_read_only::<Vec4>(false), // probe_positions
                texture_2d(TextureSampleType::Uint),     // probe_vis
            ),
        ),
    )
}

impl Plugin for Cascades2_5dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CascadesLayout>()
            .init_resource::<SunParams>()
            .add_plugins((
                ExtractResourcePlugin::<CascadesLayout>::default(),
                ExtractResourcePlugin::<SunParams>::default(),
            ))
            .add_systems(PostUpdate, update_sun);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<Relight>()
            .add_systems(bevy::render::RenderStartup, setup_pipelines)
            .add_systems(
                Render,
                prepare_buffers.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Core3d, dispatch_cascades.in_set(Core3dSystems::Prepass));
    }
}

fn update_sun(
    lights: Query<(&DirectionalLight, &GlobalTransform)>,
    cameras: Query<Option<&bevy::camera::Exposure>, With<Camera3d>>,
    mut sun: ResMut<SunParams>,
) {
    let Some((light, transform)) = lights.iter().next() else {
        return;
    };
    let exposure = cameras
        .iter()
        .next()
        .flatten()
        .copied()
        .unwrap_or_default()
        .exposure();
    let color = light.color.to_linear();
    let new = SunParams {
        toward: -transform.forward().as_vec3(),
        irradiance: Vec3::new(color.red, color.green, color.blue) * light.illuminance * exposure,
    };
    if new.toward != sun.toward || new.irradiance != sun.irradiance {
        *sun = new;
    }
}

fn setup_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let light_layout = BindGroupLayoutDescriptor::new(
        "cascades_light_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // normal_min
                texture_2d(TextureSampleType::Float { filterable: false }), // normal_max
                texture_2d(TextureSampleType::Float { filterable: false }), // albedo_min
                texture_2d(TextureSampleType::Float { filterable: false }), // albedo_max
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                storage_buffer_read_only::<Vec4>(false), // sh_probes
                uniform_buffer::<LightUniforms>(false),  // light_info
                storage_buffer_read_only::<Vec4>(false), // bounce_history_in
                storage_buffer::<Vec4>(false),           // bounce_history_out
            ),
        ),
    );
    let trace_layout = BindGroupLayoutDescriptor::new(
        "cascades_trace_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // radiance_min
                texture_2d(TextureSampleType::Float { filterable: false }), // radiance_max
                storage_buffer::<Vec4>(false),                              // rays
                storage_buffer_read_only::<Vec4>(false),                    // hits
            ),
        ),
    );
    let gather_layout = BindGroupLayoutDescriptor::new(
        "cascades_gather_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec4>(false), // rays
                storage_buffer::<Vec4>(false),           // sh_probes
                storage_buffer_read_only::<Vec4>(false), // subsamples
            ),
        ),
    );
    let march_layout = BindGroupLayoutDescriptor::new(
        "cascades_march_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            // Written once, read by every relight after that.
            (storage_buffer::<Vec4>(false),), // hits
        ),
    );
    let probe_positions_layout = BindGroupLayoutDescriptor::new(
        "cascades_probe_positions_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (storage_buffer::<Vec4>(false),),
        ),
    );
    let bake_probe_adj_layout = BindGroupLayoutDescriptor::new(
        "cascades_bake_probe_adj_layout",
        &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (storage_buffer::<u32>(false),)),
    );
    let blend_layout = BindGroupLayoutDescriptor::new(
        "cascades_blend_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec4>(false), // sh_probes_in
                storage_buffer_read_only::<Vec4>(false), // sh_probes_cur
                storage_buffer::<Vec4>(false),           // sh_probes_out
                uniform_buffer::<BlendUniforms>(false),  // blend uniforms
                storage_buffer_read_only::<u32>(false),  // probe_adj
            ),
        ),
    );

    let pipeline = |label: String,
                    group1: &BindGroupLayoutDescriptor,
                    shader: &Handle<Shader>,
                    entry: String| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![common_layout(), group1.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let light_shader = asset_server.load(LIGHT_SHADER_PATH);
    let trace_shader = asset_server.load(TRACE_SHADER_PATH);
    let gather_shader = asset_server.load(GATHER_SHADER_PATH);
    let march_shader = asset_server.load(MARCH_SHADER_PATH);
    let blend_shader = asset_server.load(BLEND_SHADER_PATH);
    let probe_positions_shader = asset_server.load(PROBE_POSITIONS_SHADER_PATH);
    let bake_probe_vis_shader = asset_server.load(BAKE_PROBE_VIS_SHADER_PATH);
    let bake_probe_adj_shader = asset_server.load(BAKE_PROBE_ADJ_SHADER_PATH);
    let bake_probe_vis_layout = BindGroupLayoutDescriptor::new(
        "cascades_bake_probe_vis_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_storage_2d(
                TextureFormat::R32Uint,
                StorageTextureAccess::WriteOnly,
            ),),
        ),
    );
    let cascades = 0..layout::CASCADES;

    commands.insert_resource(CascadesPipeline {
        _trace_library: asset_server.load(TRACE_LIBRARY_PATH),
        light_pipeline: pipeline(
            "cascades_surface_light".into(),
            &light_layout,
            &light_shader,
            "surface_light".into(),
        ),
        trace_pipelines: cascades
            .clone()
            .map(|c| {
                pipeline(
                    format!("cascades_trace{c}"),
                    &trace_layout,
                    &trace_shader,
                    format!("cascade_{c}"),
                )
            })
            .collect(),
        gather_pipelines: cascades
            .clone()
            .map(|c| {
                pipeline(
                    format!("cascades_gather{c}"),
                    &gather_layout,
                    &gather_shader,
                    format!("gather_{c}"),
                )
            })
            .collect(),
        march_pipelines: cascades
            .clone()
            .map(|c| {
                pipeline(
                    format!("cascades_march{c}"),
                    &march_layout,
                    &march_shader,
                    format!("march_{c}"),
                )
            })
            .collect(),
        probe_positions_pipelines: cascades
            .clone()
            .map(|c| {
                pipeline(
                    format!("cascades_probe_positions{c}"),
                    &probe_positions_layout,
                    &probe_positions_shader,
                    format!("probe_positions_{c}"),
                )
            })
            .collect(),
        bake_probe_vis_pipeline: pipeline(
            "cascades_bake_probe_vis".into(),
            &bake_probe_vis_layout,
            &bake_probe_vis_shader,
            "main".into(),
        ),
        bake_probe_adj_pipelines: cascades
            .map(|c| {
                pipeline(
                    format!("cascades_bake_probe_adj{c}"),
                    &bake_probe_adj_layout,
                    &bake_probe_adj_shader,
                    format!("bake_adj_{c}"),
                )
            })
            .collect(),
        blend_pipeline: pipeline(
            "cascades_blend_sh".into(),
            &blend_layout,
            &blend_shader,
            "blend_sh".into(),
        ),
        light_layout,
        trace_layout,
        gather_layout,
        march_layout,
        probe_positions_layout,
        bake_probe_adj_layout,
        blend_layout,
        bake_probe_vis_layout,
    });
}

/// Radiance a probe ray records when it reaches open sky, in the same units
/// as the sunlit surface radiance the bake stores (sun irradiance already
/// scaled by the camera exposure). Override with S_RC_SKY="r,g,b" (linear).
/// Mirrored by the main camera's clear color too, so the backdrop that
/// shows through the doorway is the same sky the probes read.
pub fn sky_color(sun_irradiance: Vec3) -> Vec4 {
    if let Ok(value) = std::env::var("S_RC_SKY") {
        let p: Vec<f32> = value
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(1.0))
            .collect();
        if p.len() >= 3 {
            return Vec4::new(p[0], p[1], p[2], 1.0);
        }
    }
    default_sky(sun_irradiance)
}

fn default_sky(sun_irradiance: Vec3) -> Vec4 {
    Vec4::new(
        sun_irradiance.x * 0.05,
        sun_irradiance.y * 0.065,
        sun_irradiance.z * 0.11,
        1.0,
    )
}

fn prepare_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    layout: Res<CascadesLayout>,
    sun: Res<SunParams>,
    heightmap: Option<Res<Heightmap>>,
    bake: Option<Res<BakeState>>,
    existing: Option<Res<CascadesRenderData>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(heightmap) = heightmap else {
        return;
    };
    let image = |handle: &Handle<Image>| gpu_images.get(handle).map(|i| &i.texture_view);
    let (
        Some(heights),
        Some(normal_min),
        Some(normal_max),
        Some(albedo_min),
        Some(albedo_max),
        Some(radiance_min),
        Some(radiance_max),
        Some(winding),
        Some(probe_vis),
    ) = (
        image(&heightmap.texture),
        image(&heightmap.metadata_min),
        image(&heightmap.metadata_max),
        image(&heightmap.albedo_min),
        image(&heightmap.albedo_max),
        image(&heightmap.radiance_min),
        image(&heightmap.radiance_max),
        image(&heightmap.winding),
        image(&heightmap.probe_vis),
    )
    else {
        return;
    };
    let layout = &layout.layout;

    let mut cascades = [CascadeInfoUniform::default(); layout::CASCADES as usize];
    for (info, c) in cascades.iter_mut().zip(&layout.cascades) {
        *info = CascadeInfoUniform {
            size: c.size,
            interval: Vec2::from(c.interval),
            ray_base: c.hit_base,
            probe_base: c.sh_base,
            spacing: c.spacing,
            octa_side: c.octa_side,
            sub: c.sub,
            sub_base: c.sub_base,
            ..default()
        };
    }
    let sky_color = sky_color(sun.irradiance);
    let uniforms = CascadesUniforms {
        hmap_min: heightmap.min_bounds,
        hmap_max: heightmap.max_bounds,
        hmap_res: heightmap.resolution,
        cascade_count: layout.cascades.len() as u32,
        hmap_top: heightmap.top,
        split: crate::heightmap::SPLIT,
        sun_toward: sun.toward,
        sun_irradiance: sun.irradiance,
        sky_color,
        cascades,
    };
    let mut bytes = encase::UniformBuffer::new(Vec::new());
    bytes.write(&uniforms).unwrap();

    let probes: u64 = layout
        .cascades
        .iter()
        .map(|c| (c.size.x * c.size.y) as u64 * layout::LAYERS as u64)
        .sum();
    let (
        rays_buffer,
        probe_positions_buffer,
        sh_probes,
        sh_probes_blend_buffer,
        sh_probes_scratch_buffer,
        sh_probes_history,
        bounce_history_buffers,
        probe_adj_buffer,
        subsamples_buffer,
        hits_buffer,
        uniform_buffer,
        light_uniform_buffers,
        blend_uniform_buffers,
    ) = match &existing {
        Some(e) => (
            e.rays_buffer.clone(),
            e.probe_positions_buffer.clone(),
            e.sh_probes.clone(),
            e.sh_probes_blend_buffer.clone(),
            e.sh_probes_scratch_buffer.clone(),
            e.sh_probes_history.clone(),
            e.bounce_history_buffers.clone(),
            e.probe_adj_buffer.clone(),
            e.subsamples_buffer.clone(),
            e.hits_buffer.clone(),
            e.uniform_buffer.clone(),
            e.light_uniform_buffers.clone(),
            e.blend_uniform_buffers.clone(),
        ),
        None => {
            let storage = |label: &'static str, vec4s: u64, usage: BufferUsages| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size: vec4s * 16,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let readback = BufferUsages::STORAGE | BufferUsages::COPY_DST;
            let subsamples: Vec<u8> = layout
                .subsamples
                .iter()
                .flat_map(|v| v.to_array())
                .flat_map(f32::to_ne_bytes)
                .collect();
            let initial_adj: Vec<u8> = (0..probes.max(16))
                .flat_map(|_| 0x1FFu32.to_ne_bytes())
                .collect();
            let light_uniform_buffers: [Buffer; MAX_BOUNCE_COUNT as usize] =
                std::array::from_fn(|i| {
                    render_device.create_buffer(&BufferDescriptor {
                        label: Some(&format!("cascades_light_uniforms_{i}")),
                        size: LightUniforms::min_size().get(),
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });
            let blend_uniform_buffers: [Buffer; MAX_BLUR_PASSES] = std::array::from_fn(|i| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("cascades_blend_uniforms_{i}")),
                    size: BlendUniforms::min_size().get(),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            (
                storage("cascades_rays", layout.hit_words, readback),
                storage(
                    "cascades_probe_positions",
                    probes * 2,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                ),
                [
                    storage("cascades_sh_probes_0", probes * SH_STRIDE, readback),
                    storage("cascades_sh_probes_1", probes * SH_STRIDE, readback),
                    storage("cascades_sh_probes_2", probes * SH_STRIDE, readback),
                ],
                storage(
                    "cascades_sh_probes_blend",
                    probes * SH_STRIDE,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                ),
                storage(
                    "cascades_sh_probes_scratch",
                    probes * SH_STRIDE,
                    BufferUsages::STORAGE,
                ),
                [
                    storage(
                        "cascades_sh_probes_hist_0",
                        probes * SH_STRIDE,
                        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    ),
                    storage(
                        "cascades_sh_probes_hist_1",
                        probes * SH_STRIDE,
                        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    ),
                ],
                [
                    storage(
                        "cascades_bounce_history_0",
                        (heightmap.resolution.x * heightmap.resolution.y) as u64 * 2,
                        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    ),
                    storage(
                        "cascades_bounce_history_1",
                        (heightmap.resolution.x * heightmap.resolution.y) as u64 * 2,
                        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    ),
                ],
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("cascades_probe_adj"),
                    contents: &initial_adj,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                }),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("cascades_subsamples"),
                    contents: &subsamples,
                    usage: BufferUsages::STORAGE,
                }),
                storage("cascades_hits", layout.hit_words * HITS_PER_RAY, readback),
                render_device.create_buffer(&BufferDescriptor {
                    label: Some("cascades_uniforms"),
                    size: bytes.as_ref().len() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                light_uniform_buffers,
                blend_uniform_buffers,
            )
        }
    };
    // The sun moves every frame, so this is a write rather than the
    // allocation it used to be.
    render_queue.write_buffer(&uniform_buffer, 0, bytes.as_ref());

    // Once the bake is confirmed, every image these bind groups name has
    // settled: the heightmap and the winding map are written exactly once
    // (and only then, while the bake is still unverified), and the rest are
    // written by compute without ever being re-uploaded. Keeping the previous
    // frame's groups avoids four allocations and four rounds of wgpu
    // validation per frame.
    if existing.is_some() && bake.as_ref().is_some_and(|b| b.is_verified()) {
        return;
    }

    let bind_group =
        |label: &str, layout: &BindGroupLayoutDescriptor, entries: &[BindGroupEntry]| {
            render_device.create_bind_group(
                label,
                &pipeline_cache.get_bind_group_layout(layout),
                entries,
            )
        };
    let common_bind_group = bind_group(
        "cascades_common",
        &common_layout(),
        &BindGroupEntries::sequential((
            heights,
            uniform_buffer.as_entire_binding(),
            normal_min,
            normal_max,
            winding,
            probe_positions_buffer.as_entire_binding(),
            probe_vis,
        )),
    );
    let probe_vis_dummy = render_device.create_texture(&TextureDescriptor {
        label: Some("cascades_probe_vis_dummy"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Uint,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let probe_vis_dummy_view = probe_vis_dummy.create_view(&TextureViewDescriptor::default());

    let probe_positions_dummy_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("cascades_probe_positions_dummy"),
        size: 16,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let probe_positions_pass_bind_group = bind_group(
        "cascades_probe_positions_pass",
        &common_layout(),
        &BindGroupEntries::sequential((
            heights,
            uniform_buffer.as_entire_binding(),
            normal_min,
            normal_max,
            winding,
            probe_positions_dummy_buffer.as_entire_binding(),
            &probe_vis_dummy_view,
        )),
    );
    let probe_vis_pass_bind_group = bind_group(
        "cascades_probe_vis_pass",
        &common_layout(),
        &BindGroupEntries::sequential((
            heights,
            uniform_buffer.as_entire_binding(),
            normal_min,
            normal_max,
            winding,
            probe_positions_buffer.as_entire_binding(),
            &probe_vis_dummy_view,
        )),
    );
    let light_bind_groups: [[[BindGroup; 3]; MAX_BOUNCE_COUNT as usize]; 2] =
        std::array::from_fn(|bh| {
            std::array::from_fn(|b| {
                [0, 1, 2].map(|p| {
                    // b == 0 reads the converged/blurred probe buffer from the previous frame for temporal bounce.
                    // b > 0 reads the fresh in-frame gather target of phase p.
                    let probe_buf = if b == 0 {
                        &sh_probes_blend_buffer
                    } else {
                        &sh_probes[G_OF_PHASE[p]]
                    };
                    bind_group(
                        &format!("cascades_light_h{bh}_b{b}_p{p}"),
                        &pipeline.light_layout,
                        &BindGroupEntries::sequential((
                            normal_min,
                            normal_max,
                            albedo_min,
                            albedo_max,
                            radiance_min,
                            radiance_max,
                            probe_buf.as_entire_binding(),
                            light_uniform_buffers[b].as_entire_binding(),
                            bounce_history_buffers[bh].as_entire_binding(),
                            bounce_history_buffers[1 - bh].as_entire_binding(),
                        )),
                    )
                })
            })
        });
    let trace_bind_group = bind_group(
        "cascades_trace",
        &pipeline.trace_layout,
        &BindGroupEntries::sequential((
            radiance_min,
            radiance_max,
            rays_buffer.as_entire_binding(),
            hits_buffer.as_entire_binding(),
        )),
    );
    // One gather and one blend bind group per rotation phase. The gather
    // writes the phase's G (next fresh relight); the blend reads the phase's
    // B (prev) and A (cur) into the cross-fade buffer. Only the phase binds
    // differ, so the dispatch just selects the right set for Relight::phase.
    let gather_bind_groups = [0, 1, 2].map(|p| {
        bind_group(
            [
                "cascades_gather_p0",
                "cascades_gather_p1",
                "cascades_gather_p2",
            ][p],
            &pipeline.gather_layout,
            &BindGroupEntries::sequential((
                rays_buffer.as_entire_binding(),
                sh_probes[G_OF_PHASE[p]].as_entire_binding(),
                subsamples_buffer.as_entire_binding(),
            )),
        )
    });
    let march_bind_group = bind_group(
        "cascades_march",
        &pipeline.march_layout,
        &BindGroupEntries::sequential((hits_buffer.as_entire_binding(),)),
    );
    let probe_positions_bind_group = bind_group(
        "cascades_probe_positions",
        &pipeline.probe_positions_layout,
        &BindGroupEntries::sequential((probe_positions_buffer.as_entire_binding(),)),
    );
    let probe_vis_bind_group = bind_group(
        "cascades_probe_vis",
        &pipeline.bake_probe_vis_layout,
        &BindGroupEntries::sequential((probe_vis,)),
    );
    let probe_adj_bind_group = bind_group(
        "cascades_bake_probe_adj",
        &pipeline.bake_probe_adj_layout,
        &BindGroupEntries::sequential((probe_adj_buffer.as_entire_binding(),)),
    );
    let blend_temporal_bind_groups = [0, 1].map(|h| {
        [0, 1, 2].map(|p| {
            bind_group(
                &format!("cascades_blend_temporal_h{h}_p{p}"),
                &pipeline.blend_layout,
                &BindGroupEntries::sequential((
                    sh_probes_history[h].as_entire_binding(),
                    sh_probes[A_OF_PHASE[p]].as_entire_binding(),
                    sh_probes_history[1 - h].as_entire_binding(),
                    blend_uniform_buffers[0].as_entire_binding(),
                    probe_adj_buffer.as_entire_binding(),
                )),
            )
        })
    });
    let blend_spatial_p1_to_scratch = [0, 1].map(|h_curr| {
        bind_group(
            &format!("cascades_spatial_p1_s_h{h_curr}"),
            &pipeline.blend_layout,
            &BindGroupEntries::sequential((
                sh_probes_history[h_curr].as_entire_binding(),
                sh_probes_history[h_curr].as_entire_binding(),
                sh_probes_scratch_buffer.as_entire_binding(),
                blend_uniform_buffers[1].as_entire_binding(),
                probe_adj_buffer.as_entire_binding(),
            )),
        )
    });
    let blend_spatial_p1_to_blend = [0, 1].map(|h_curr| {
        bind_group(
            &format!("cascades_spatial_p1_b_h{h_curr}"),
            &pipeline.blend_layout,
            &BindGroupEntries::sequential((
                sh_probes_history[h_curr].as_entire_binding(),
                sh_probes_history[h_curr].as_entire_binding(),
                sh_probes_blend_buffer.as_entire_binding(),
                blend_uniform_buffers[1].as_entire_binding(),
                probe_adj_buffer.as_entire_binding(),
            )),
        )
    });
    let blend_ping: [BindGroup; MAX_BLUR_PASSES] = std::array::from_fn(|k| {
        bind_group(
            "cascades_blend_ping",
            &pipeline.blend_layout,
            &BindGroupEntries::sequential((
                sh_probes_scratch_buffer.as_entire_binding(),
                sh_probes_scratch_buffer.as_entire_binding(),
                sh_probes_blend_buffer.as_entire_binding(),
                blend_uniform_buffers[k].as_entire_binding(),
                probe_adj_buffer.as_entire_binding(),
            )),
        )
    });
    let blend_pong: [BindGroup; MAX_BLUR_PASSES] = std::array::from_fn(|k| {
        bind_group(
            "cascades_blend_pong",
            &pipeline.blend_layout,
            &BindGroupEntries::sequential((
                sh_probes_blend_buffer.as_entire_binding(),
                sh_probes_blend_buffer.as_entire_binding(),
                sh_probes_scratch_buffer.as_entire_binding(),
                blend_uniform_buffers[k].as_entire_binding(),
                probe_adj_buffer.as_entire_binding(),
            )),
        )
    });

    commands.insert_resource(CascadesRenderData {
        rays_buffer,
        probe_positions_buffer,
        sh_probes,
        sh_probes_blend_buffer,
        sh_probes_scratch_buffer,
        sh_probes_history,
        bounce_history_buffers,
        probe_adj_buffer,
        hits_buffer,
        uniform_buffer,
        light_uniform_buffers,
        blend_uniform_buffers,
        subsamples_buffer,
        common_bind_group,
        probe_positions_pass_bind_group,
        probe_positions_bind_group,
        probe_vis_pass_bind_group,
        probe_vis_bind_group,
        probe_adj_bind_group,
        light_bind_groups,
        trace_bind_group,
        gather_bind_groups,
        march_bind_group,
        blend_temporal_bind_groups,
        blend_spatial_p1_to_scratch,
        blend_spatial_p1_to_blend,
        blend_ping,
        blend_pong,
    });
}

#[derive(Default)]
struct CascadeLocals {
    frame: u32,
    marched: bool,
    bakes_done: (bool, bool, bool),
    was_indirect: bool,
    hist_index: usize,
    history_valid: bool,
    bounce_hist_index: usize,
    bounce_history_valid: bool,
    prev_temporal_bounce: u32,
    prev_temporal_blur: u32,
    prev_in_frame_bounces: u32,
}

fn dispatch_cascades(
    layout: Res<CascadesLayout>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    heightmap: Option<Res<Heightmap>>,
    render_data: Option<Res<CascadesRenderData>>,
    bake: Option<Res<BakeState>>,
    mut relight: ResMut<Relight>,
    render_queue: Res<RenderQueue>,
    mut render_context: RenderContext,
    mut locals: Local<CascadeLocals>,
    blend_toggle: Res<BlendToggleState>,
    lighting_state: Option<Res<LightingState>>,
) {
    let (Some(render_data), Some(heightmap)) = (render_data, heightmap) else {
        return;
    };
    // Every pass reads the previous one's output; skip the frame until all
    // have compiled.
    let get = |id: &CachedComputePipelineId| pipeline_cache.get_compute_pipeline(*id);
    let (
        Some(light),
        Some(traces),
        Some(gathers),
        Some(marches),
        Some(probe_positions),
        Some(bake_probe_vis),
        Some(bake_probe_adj),
        Some(blend),
    ) = (
        get(&pipeline.light_pipeline),
        pipeline
            .trace_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        pipeline
            .gather_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        pipeline
            .march_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        pipeline
            .probe_positions_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        get(&pipeline.bake_probe_vis_pipeline),
        pipeline
            .bake_probe_adj_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        get(&pipeline.blend_pipeline),
    )
    else {
        return;
    };

    // The march depends only on the scene, so it happens once. A bake that has
    // to retry rewrites the heightmap underneath it, so an unverified map
    // invalidates the stored hits and sends the march back through.
    let verified = bake.as_ref().is_some_and(|b| b.is_verified());
    let need_march = !locals.marched;
    if need_march && !verified {
        return;
    }

    let indirect_enabled = lighting_state.as_ref().map_or(true, |l| l.indirect_enabled);
    if locals.marched && !indirect_enabled {
        locals.was_indirect = false;
        return;
    }

    let interval = relight.interval.max(1);
    let since = locals.frame % interval;
    locals.frame = locals.frame.wrapping_add(1);
    relight.alpha = if interval == 1 {
        1.0
    } else {
        (since as f32 + 1.0) / interval as f32
    };
    let re_enabled = indirect_enabled && !locals.was_indirect;
    locals.was_indirect = indirect_enabled;

    let n_bounce = TEMPORAL_BOUNCE_PASSES
        .load(Ordering::Relaxed)
        .clamp(1, MAX_TEMPORAL_BOUNCE_PASSES);
    let bounce_changed = locals.prev_temporal_bounce != n_bounce;
    locals.prev_temporal_bounce = n_bounce;

    let n_blur = TEMPORAL_BLUR_PASSES
        .load(Ordering::Relaxed)
        .clamp(1, MAX_TEMPORAL_PASSES);
    let blur_changed = locals.prev_temporal_blur != n_blur;
    locals.prev_temporal_blur = n_blur;

    let in_frame_bounces = BOUNCE_COUNT
        .load(Ordering::Relaxed)
        .clamp(1, MAX_BOUNCE_COUNT);
    let in_bounces_changed = locals.prev_in_frame_bounces != in_frame_bounces;
    locals.prev_in_frame_bounces = in_frame_bounces;

    let relight_now = need_march
        || since == 0
        || re_enabled
        || bounce_changed
        || blur_changed
        || in_bounces_changed;

    if verified && !locals.bakes_done.0 {
        let mut pos_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cascades_probe_positions"),
                    ..default()
                });
        pos_pass.set_bind_group(0, &render_data.probe_positions_pass_bind_group, &[]);
        pos_pass.set_bind_group(1, &render_data.probe_positions_bind_group, &[]);
        for (c, pos) in layout.layout.cascades.iter().zip(&probe_positions) {
            pos_pass.set_pipeline(pos);
            pos_pass.dispatch_workgroups(
                (c.size.x * c.size.y * layout::LAYERS).div_ceil(WORKGROUP),
                1,
                1,
            );
        }
        locals.bakes_done.0 = true;
    }

    if verified && locals.bakes_done.0 && !locals.bakes_done.1 {
        let mut vis_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cascades_bake_probe_vis"),
                    ..default()
                });
        vis_pass.set_pipeline(bake_probe_vis);
        vis_pass.set_bind_group(0, &render_data.probe_vis_pass_bind_group, &[]);
        vis_pass.set_bind_group(1, &render_data.probe_vis_bind_group, &[]);
        vis_pass.dispatch_workgroups(
            heightmap.resolution.x.div_ceil(16),
            heightmap.resolution.y.div_ceil(16),
            1,
        );
        locals.bakes_done.1 = true;
    }

    if verified && locals.bakes_done.0 && !locals.bakes_done.2 {
        let mut adj_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cascades_bake_probe_adj"),
                    ..default()
                });
        adj_pass.set_bind_group(0, &render_data.common_bind_group, &[]);
        adj_pass.set_bind_group(1, &render_data.probe_adj_bind_group, &[]);
        for (c, adj_pipe) in layout.layout.cascades.iter().zip(&bake_probe_adj) {
            adj_pass.set_pipeline(adj_pipe);
            adj_pass.dispatch_workgroups(
                (c.size.x * c.size.y * layout::LAYERS).div_ceil(WORKGROUP),
                1,
                1,
            );
        }
        locals.bakes_done.2 = true;
    }

    {
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cascades2_5d"),
                    ..default()
                });
        pass.set_bind_group(0, &render_data.common_bind_group, &[]);

        if need_march {
            pass.set_bind_group(1, &render_data.march_bind_group, &[]);
            for (c, march) in layout.layout.cascades.iter().zip(&marches).rev() {
                pass.set_pipeline(march);
                pass.dispatch_workgroups((c.rays() as u32).div_ceil(WORKGROUP), 1, 1);
            }
            locals.marched = true;
        }

        if !indirect_enabled {
            return;
        }

        if relight_now {
            let phase = (relight.phase % 3) as usize;
            let bounces = in_frame_bounces;
            let use_temporal = blend_toggle.temporal_bounce && !need_march;

            let bh_prev = locals.bounce_hist_index;
            let bh_curr = 1 - bh_prev;
            locals.bounce_hist_index = bh_curr;

            let temporal_weight = if !use_temporal || n_bounce <= 1 {
                0.0
            } else {
                ((n_bounce - 1) as f32 / (MAX_TEMPORAL_BOUNCE_PASSES - 1) as f32).powf(0.6)
            };
            if use_temporal {
                locals.bounce_history_valid = true;
            } else {
                locals.bounce_history_valid = false;
            }

            for b in 0..bounces {
                let mut light_bytes = encase::UniformBuffer::new(Vec::new());
                light_bytes
                    .write(&LightUniforms {
                        bounce_index: b,
                        temporal_bounce: if b == 0 && use_temporal && n_bounce > 1 {
                            1
                        } else {
                            0
                        },
                        temporal_weight,
                        pad0: 0,
                    })
                    .unwrap();
                render_queue.write_buffer(
                    &render_data.light_uniform_buffers[b as usize],
                    0,
                    light_bytes.as_ref(),
                );

                pass.set_pipeline(light);
                pass.set_bind_group(
                    1,
                    &render_data.light_bind_groups[bh_prev][b as usize][phase],
                    &[],
                );
                pass.dispatch_workgroups(
                    heightmap.resolution.x.div_ceil(LIGHT_WORKGROUP),
                    heightmap.resolution.y.div_ceil(LIGHT_WORKGROUP),
                    1,
                );

                pass.set_bind_group(1, &render_data.trace_bind_group, &[]);
                for (c, trace) in layout.layout.cascades.iter().zip(&traces).rev() {
                    pass.set_pipeline(trace);
                    pass.dispatch_workgroups((c.rays() as u32).div_ceil(WORKGROUP), 1, 1);
                }

                pass.set_bind_group(1, &render_data.gather_bind_groups[phase], &[]);
                for (c, gather) in layout.layout.cascades.iter().zip(&gathers) {
                    pass.set_pipeline(gather);
                    pass.dispatch_workgroups(
                        (c.size.x * c.size.y * layout::LAYERS).div_ceil(WORKGROUP),
                        1,
                        1,
                    );
                }
            }

            relight.phase = (relight.phase + 1).rem_euclid(3);
        }
    }

    // Number of spatial blur iterations applied (0 if lowpass is disabled).
    let spatial_passes = if blend_toggle.lowpass {
        LOWPASS_BLUR_PASSES
            .load(Ordering::Relaxed)
            .min((MAX_BLUR_PASSES - 1) as u32)
    } else {
        0
    };

    let probe_words = layout
        .layout
        .cascades
        .iter()
        .map(|c| (c.size.x * c.size.y) as u64 * layout::LAYERS as u64)
        .sum::<u64>()
        * SH_STRIDE;

    let cross_layer_radius = CROSS_LAYER_RADIUS_TENTHS.load(Ordering::Relaxed) as f32 / 10.0;

    let hist_prev = locals.hist_index;
    let hist_curr = 1 - hist_prev;
    locals.hist_index = hist_curr;

    let n_temporal = TEMPORAL_BLUR_PASSES
        .load(Ordering::Relaxed)
        .clamp(1, MAX_TEMPORAL_PASSES);
    let is_first_frame = need_march || re_enabled || !locals.history_valid;
    let lambda = if is_first_frame {
        1.0
    } else {
        1.0 / (n_temporal as f32)
    };
    locals.history_valid = true;

    let blend_alpha = if interval > 1 { relight.alpha } else { lambda };

    // Pass 0 uniform: strictly per-probe temporal EMA accumulation (lowpass = 0, alpha = blend_alpha)
    let mut bytes0 = encase::UniformBuffer::new(Vec::new());
    bytes0
        .write(&BlendUniforms {
            alpha: blend_alpha,
            probe_words: probe_words as u32,
            lowpass: 0,
            pass_index: 0,
            cross_layer_radius: 0.0,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        })
        .unwrap();
    render_queue.write_buffer(&render_data.blend_uniform_buffers[0], 0, bytes0.as_ref());

    // Uniforms for spatial blur passes 1..=spatial_passes (lowpass = 1)
    for k in 1..=spatial_passes as usize {
        let mut bytes = encase::UniformBuffer::new(Vec::new());
        bytes
            .write(&BlendUniforms {
                alpha: 0.0,
                probe_words: probe_words as u32,
                lowpass: 1,
                pass_index: k as u32,
                cross_layer_radius,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            })
            .unwrap();
        render_queue.write_buffer(&render_data.blend_uniform_buffers[k], 0, bytes.as_ref());
    }

    let phase = (relight.phase % 3) as usize;

    // --- Pass 0: Pure temporal EMA accumulation across frames (t and t-1, per probe, zero spatial neighbor reads) ---
    {
        let mut blend_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cascades_blend_sh_temporal"),
                    ..default()
                });
        blend_pass.set_bind_group(0, &render_data.common_bind_group, &[]);
        blend_pass.set_pipeline(blend);
        blend_pass.set_bind_group(
            1,
            &render_data.blend_temporal_bind_groups[hist_prev][phase],
            &[],
        );
        blend_pass.dispatch_workgroups(probe_words.div_ceil(BLEND_WORKGROUP as u64) as u32, 1, 1);
    }

    if spatial_passes == 0 {
        // Pure temporal EMA: copy history_curr directly to sh_probes_blend_buffer
        let probe_bytes = probe_words * 16;
        render_context.command_encoder().copy_buffer_to_buffer(
            &render_data.sh_probes_history[hist_curr],
            0,
            &render_data.sh_probes_blend_buffer,
            0,
            probe_bytes,
        );
    } else {
        // Spatial blur passes 1..=spatial_passes
        let p1_writes_to_blend = spatial_passes % 2 != 0;
        {
            let mut blend_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cascades_blend_sh_spatial_p1"),
                        ..default()
                    });
            blend_pass.set_bind_group(0, &render_data.common_bind_group, &[]);
            blend_pass.set_pipeline(blend);
            if p1_writes_to_blend {
                blend_pass.set_bind_group(
                    1,
                    &render_data.blend_spatial_p1_to_blend[hist_curr],
                    &[],
                );
            } else {
                blend_pass.set_bind_group(
                    1,
                    &render_data.blend_spatial_p1_to_scratch[hist_curr],
                    &[],
                );
            }
            blend_pass.dispatch_workgroups(
                probe_words.div_ceil(BLEND_WORKGROUP as u64) as u32,
                1,
                1,
            );
        }

        let mut writing_to_blend = p1_writes_to_blend;
        for k in 2..=spatial_passes as usize {
            writing_to_blend = !writing_to_blend;
            let mut blend_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cascades_blend_sh_spatial_pn"),
                        ..default()
                    });
            blend_pass.set_bind_group(0, &render_data.common_bind_group, &[]);
            blend_pass.set_pipeline(blend);
            if writing_to_blend {
                blend_pass.set_bind_group(1, &render_data.blend_ping[k], &[]);
            } else {
                blend_pass.set_bind_group(1, &render_data.blend_pong[k], &[]);
            }
            blend_pass.dispatch_workgroups(
                probe_words.div_ceil(BLEND_WORKGROUP as u64) as u32,
                1,
                1,
            );
        }
    }
}
