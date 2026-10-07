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
use crate::ui::ParallaxState;

const TRACE_LIBRARY_PATH: &str = "shaders/heightmap_trace.wgsl";
const LIGHT_SHADER_PATH: &str = "shaders/surface_light.wgsl";
const TRACE_SHADER_PATH: &str = "shaders/cascades2_5d.wgsl";
const GATHER_SHADER_PATH: &str = "shaders/gather_cascades.wgsl";
const MARCH_SHADER_PATH: &str = "shaders/march_hits.wgsl";
const BLEND_SHADER_PATH: &str = "shaders/blend_sh.wgsl";
/// Workgroup sizes: 8x8 texels for the light pass, 64 rays or probes for
/// the others.
const LIGHT_WORKGROUP: u32 = 8;
const WORKGROUP: u32 = 64;
/// SH probe stride in vec4s: 9 coefficients, then (position, radius).
pub const SH_STRIDE: u64 = 10;
/// Slots per ray in the one-shot hit cache: the top cascade's own trace, or
/// the four segment traces the cascades below it use instead.
const HITS_PER_RAY: u64 = 4;
/// Frames between two relights. The sun moves every frame, so this is the
/// period the indirect light lags behind the direct light by.
const RELIGHT_INTERVAL: u32 = 4;
/// Which of the three rotating probe buffers plays each role, per
/// [`Relight::phase`]. `A` is `blend_sh`'s `cur` operand (weight `alpha`, the
/// freshest relight), `B` its `prev` operand (weight `1 - alpha`), and `G`
/// the buffer the next gather overwrites. A relight therefore lands on the
/// gather target, becomes `A` at weight 0 on that same frame, climbs with
/// `alpha` through the interval, and hits weight 1 "by itself" on the next
/// sync, when the swap turns it into `B`.
const A_OF_PHASE: [usize; 3] = [0, 2, 1];
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
        Self {
            interval: RELIGHT_INTERVAL,
            alpha: 0.0,
            phase: 0,
        }
    }
}

/// What `blend_sh` mixes with. Sized to a whole 16 bytes on purpose: a
/// two-scalar struct is 8 bytes naturally, but uniform address space rounds a
/// struct up to 16, and the two sides have to agree on the binding's minimum
/// size or the bind group fails validation.
#[derive(ShaderType, Clone, Copy, Default)]
struct BlendUniforms {
    pub alpha: f32,
    pub probe_words: u32,
    pub pad0: u32,
    pub pad1: u32,
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
    pub parallax_endpoint: u32,
    pub parallax_pad: u32,
}

#[derive(Resource)]
struct CascadesPipeline {
    /// Keeps the shared WGSL module loaded so the passes can import it.
    _trace_library: Handle<Shader>,
    light_layout: BindGroupLayoutDescriptor,
    trace_layout: BindGroupLayoutDescriptor,
    gather_layout: BindGroupLayoutDescriptor,
    march_layout: BindGroupLayoutDescriptor,
    blend_layout: BindGroupLayoutDescriptor,
    light_pipeline: CachedComputePipelineId,
    trace_pipelines: Vec<CachedComputePipelineId>,
    gather_pipelines: Vec<CachedComputePipelineId>,
    march_pipelines: Vec<CachedComputePipelineId>,
    blend_pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct CascadesRenderData {
    pub rays_buffer: Buffer,
    /// Three rotating SH probe buffers. Which holds the freshest relight, which
    /// the last one and which is the next gather target follows [`Relight::phase`]
    /// through [`A_OF_PHASE`], [`B_OF_PHASE`] and [`G_OF_PHASE`]; at each relight
    /// `G` becomes `A`, `A` becomes `B` and `B` becomes `G`, so the fresh relight
    /// fades in over the interval while the old one fades out - no copy.
    pub sh_probes: [Buffer; 3],
    /// What `shade` reads: the cross-fade between the two relights, written
    /// once per frame over the whole probe set.
    pub sh_probes_blend_buffer: Buffer,
    /// Where each ray landed, written once by `march_hits.wgsl`.
    pub hits_buffer: Buffer,
    /// Persistent so a moving sun rewrites its contents instead of allocating
    /// a new buffer every frame.
    pub uniform_buffer: Buffer,
    /// `BlendUniforms`, rewritten every frame.
    blend_uniform_buffer: Buffer,
    subsamples_buffer: Buffer,
    /// Group 0 of every pass: the heightmap and the uniforms.
    pub common_bind_group: BindGroup,
    light_bind_group: BindGroup,
    trace_bind_group: BindGroup,
    /// One per rotation phase, writing into the phase's gather target.
    gather_bind_groups: [BindGroup; 3],
    march_bind_group: BindGroup,
    /// One per rotation phase, reading the phase's `prev` and `cur` buffers.
    blend_bind_groups: [BindGroup; 3],
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
    let blend_layout = BindGroupLayoutDescriptor::new(
        "cascades_blend_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec4>(false), // sh_probes_prev
                storage_buffer_read_only::<Vec4>(false), // sh_probes
                storage_buffer::<Vec4>(false),           // sh_probes_blend
                uniform_buffer::<BlendUniforms>(false),  // alpha
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
            .map(|c| {
                pipeline(
                    format!("cascades_march{c}"),
                    &march_layout,
                    &march_shader,
                    format!("march_{c}"),
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
        blend_layout,
    });
}

/// Radiance a probe ray records when it reaches open sky, in the same units
/// as the sunlit surface radiance the bake stores (sun irradiance already
/// scaled by the camera exposure). Override with S_RC_SKY="r,g,b" (linear).
fn sky_color(sun_irradiance: Vec3) -> Vec4 {
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
    parallax: Res<ParallaxState>,
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
    ) = (
        image(&heightmap.texture),
        image(&heightmap.metadata_min),
        image(&heightmap.metadata_max),
        image(&heightmap.albedo_min),
        image(&heightmap.albedo_max),
        image(&heightmap.radiance_min),
        image(&heightmap.radiance_max),
        image(&heightmap.winding),
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
        parallax_endpoint: parallax.enabled as u32,
        parallax_pad: 0,
    };
    let mut bytes = encase::UniformBuffer::new(Vec::new());
    bytes.write(&uniforms).unwrap();

    let probes: u64 = layout
        .cascades
        .iter()
        .map(|c| (c.size.x * c.size.y) as u64)
        .sum();
    let (
        rays_buffer,
        sh_probes,
        sh_probes_blend_buffer,
        subsamples_buffer,
        hits_buffer,
        uniform_buffer,
        blend_uniform_buffer,
    ) = match &existing {
            Some(e) => (
                e.rays_buffer.clone(),
                e.sh_probes.clone(),
                e.sh_probes_blend_buffer.clone(),
                e.subsamples_buffer.clone(),
                e.hits_buffer.clone(),
                e.uniform_buffer.clone(),
                e.blend_uniform_buffer.clone(),
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
                (
                    storage("cascades_rays", layout.hit_words, readback),
                    // Three rotating buffers: the roles (fresh/prev/gather)
                    // follow Relight::phase, so none has a fixed meaning. The
                    // gather overwrites one of them as `read_write`; the blend
                    // reads two of them `read`. No COPY_SRC: there is no copy
                    // anymore, the rotation moves the data by changing which
                    // bind group the passes use.
                    [
                        storage("cascades_sh_probes_0", probes * SH_STRIDE, readback),
                        storage("cascades_sh_probes_1", probes * SH_STRIDE, readback),
                        storage("cascades_sh_probes_2", probes * SH_STRIDE, readback),
                    ],
                    // Written by `blend_sh`, read by `shade`: no copy usage,
                    // it is never uploaded or read back.
                    storage("cascades_sh_probes_blend", probes * SH_STRIDE, BufferUsages::STORAGE),
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
                    // Zero-filled, which is what `blend_sh` wants before
                    // `dispatch_cascades` has written a frame: alpha 0 reads
                    // the previous relight.
                    render_device.create_buffer(&BufferDescriptor {
                        label: Some("cascades_blend_uniforms"),
                        size: BlendUniforms::min_size().get(),
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
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
        |label: &'static str, layout: &BindGroupLayoutDescriptor, entries: &[BindGroupEntry]| {
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
        )),
    );
    let light_bind_group = bind_group(
        "cascades_light",
        &pipeline.light_layout,
        &BindGroupEntries::sequential((
            normal_min,
            normal_max,
            albedo_min,
            albedo_max,
            radiance_min,
            radiance_max,
        )),
    );
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
            ["cascades_gather_p0", "cascades_gather_p1", "cascades_gather_p2"][p],
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
    let blend_bind_groups = [0, 1, 2].map(|p| {
        bind_group(
            ["cascades_blend_p0", "cascades_blend_p1", "cascades_blend_p2"][p],
            &pipeline.blend_layout,
            &BindGroupEntries::sequential((
                sh_probes[B_OF_PHASE[p]].as_entire_binding(),
                sh_probes[A_OF_PHASE[p]].as_entire_binding(),
                sh_probes_blend_buffer.as_entire_binding(),
                blend_uniform_buffer.as_entire_binding(),
            )),
        )
    });

    commands.insert_resource(CascadesRenderData {
        rays_buffer,
        sh_probes,
        sh_probes_blend_buffer,
        hits_buffer,
        uniform_buffer,
        blend_uniform_buffer,
        subsamples_buffer,
        common_bind_group,
        light_bind_group,
        trace_bind_group,
        gather_bind_groups,
        march_bind_group,
        blend_bind_groups,
    });
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
    mut frame: Local<u32>,
    mut marched: Local<bool>,
) {
    let (Some(render_data), Some(heightmap)) = (render_data, heightmap) else {
        return;
    };
    // Every pass reads the previous one's output; skip the frame until all
    // have compiled.
    let get = |id: &CachedComputePipelineId| pipeline_cache.get_compute_pipeline(*id);
    let (Some(light), Some(traces), Some(gathers), Some(marches), Some(blend)) = (
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
        get(&pipeline.blend_pipeline),
    ) else {
        return;
    };

    // The march depends only on the scene, so it happens once. A bake that has
    // to retry rewrites the heightmap underneath it, so an unverified map
    // invalidates the stored hits and sends the march back through.
    let verified = bake.as_ref().is_some_and(|b| b.is_verified());
    let need_march = !*marched;
    if need_march && !verified {
        return;
    }

    // Which frame of the relight interval this is, counted from 0 on the frame
    // the probes are rewritten. The march counts as one: it is the first write
    // after verification, and relighting before it has anywhere to land would
    // gather against an empty hit cache.
    let interval = relight.interval.max(1);
    let since = *frame % interval;
    *frame = frame.wrapping_add(1);
    relight.alpha = since as f32 / interval as f32;
    let relight_now = need_march || since == 0;

    // `blend_sh` runs every frame whatever else does, so its alpha is written
    // unconditionally - the cross-fade is the only thing `shade` can read.
    let probe_words = layout
        .layout
        .cascades
        .iter()
        .map(|c| (c.size.x * c.size.y) as u64)
        .sum::<u64>()
        * SH_STRIDE;
    let mut bytes = encase::UniformBuffer::new(Vec::new());
    bytes
        .write(&BlendUniforms {
            alpha: relight.alpha,
            probe_words: probe_words as u32,
            ..default()
        })
        .unwrap();
    render_queue.write_buffer(&render_data.blend_uniform_buffer, 0, bytes.as_ref());

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("cascades2_5d"),
            ..default()
        });
    pass.set_bind_group(0, &render_data.common_bind_group, &[]);

    if need_march {
        // Where every ray lands, computed here and never again while the sun
        // moves. Top-down so a cascade's segments exist before the one below
        // reads them as parents.
        pass.set_bind_group(1, &render_data.march_bind_group, &[]);
        for (c, march) in layout.layout.cascades.iter().zip(&marches).rev() {
            pass.set_pipeline(march);
            pass.dispatch_workgroups((c.rays() as u32).div_ceil(WORKGROUP), 1, 1);
        }
        *marched = true;
    }

    if relight_now {
        // The rotation phase that just gathered. The gather overwrites this
        // phase's `G` buffer; advancing the phase afterwards makes that buffer
        // the new `A` (the `cur` `blend_sh` fades in at weight `alpha`), the
        // old `A` the new `B` (fading out at `1 - alpha`), and the old `B` the
        // next gather target. The fresh relight therefore enters at weight 0
        // on its own frame and reaches weight 1 "by itself" at the next sync,
        // when the swap has made it `prev` - the exact weights the copy
        // produced, without the copy.
        let phase = (relight.phase % 3) as usize;
        pass.set_pipeline(light);
        pass.set_bind_group(1, &render_data.light_bind_group, &[]);
        pass.dispatch_workgroups(
            heightmap.resolution.x.div_ceil(LIGHT_WORKGROUP),
            heightmap.resolution.y.div_ceil(LIGHT_WORKGROUP),
            1,
        );

        // Top-down: each cascade merges the one above it.
        pass.set_bind_group(1, &render_data.trace_bind_group, &[]);
        for (c, trace) in layout.layout.cascades.iter().zip(&traces).rev() {
            pass.set_pipeline(trace);
            pass.dispatch_workgroups((c.rays() as u32).div_ceil(WORKGROUP), 1, 1);
        }

        pass.set_bind_group(1, &render_data.gather_bind_groups[phase], &[]);
        for (c, gather) in layout.layout.cascades.iter().zip(&gathers) {
            pass.set_pipeline(gather);
            pass.dispatch_workgroups((c.size.x * c.size.y).div_ceil(WORKGROUP), 1, 1);
        }

        // G becomes A, A becomes B, B becomes G.
        relight.phase = (relight.phase + 1).rem_euclid(3);
    }

    // A relight's blend runs against the phase its gather just rotated into,
    // so it reads the fresh relight as `cur` at weight `alpha` and the last
    // one as `prev` at `1 - alpha`.
    let phase = (relight.phase % 3) as usize;
    pass.set_pipeline(blend);
    pass.set_bind_group(1, &render_data.blend_bind_groups[phase], &[]);
    pass.dispatch_workgroups(probe_words.div_ceil(WORKGROUP as u64) as u32, 1, 1);
}
