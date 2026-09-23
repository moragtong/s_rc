//! 3D radiance cascades over the SDF clipmap, in world space, in
//! camera-centred windows (see layout.rs for the parameters and
//! cascades3d.wgsl for the passes). Direct light only for now.
//!
//! They run while the SDF view shows lighting (`N` to mode 3); `K` rebuilds
//! the clipmap and also throws away the cached hits.

pub mod layout;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    diagnostic::{DiagnosticPath, DiagnosticsStore},
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        diagnostic::RecordDiagnostics,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_3d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use crate::clipmap::{ClipmapCamera, ClipmapRenderSystems, ClipmapState, ClipmapTextures, RESOLUTION, TIER_VOXEL, TIERS};
use crate::sdf_view::{SceneResource, SdfView};
use layout::{CASCADES, INTERVAL0, Layout, OCTA_SIDE0, SPACING0, octa_side, rays_per_probe};

const SHADER: &str = "shaders/cascades3d.wgsl";
const TRACE_SPAN: &str = "cascades3d_trace";
pub const MAX_LIGHTS: usize = 16;
/// First SDF view mode that shows lighting (exports radiance).
pub const LIGHTING_MODE: u32 = 3;
/// SDF view mode that shows light carried by the cascades (and runs them).
pub const CASCADE_MODE: u32 = 4;

pub struct Cascades3dPlugin;

/// Render-world ordering: the cascades run after the clipmap is composed and
/// before anything that shades from them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cascades3dRenderSystems;

#[derive(ShaderType, Clone, Copy, Default)]
pub struct ClipmapUniform {
    pub origins: [IVec4; TIERS],
    pub voxels: Vec4,
    pub resolution: u32,
}

impl ClipmapUniform {
    pub fn from_state(state: &ClipmapState) -> Self {
        Self {
            origins: std::array::from_fn(|t| state.origins[t].unwrap_or_default().extend(0)),
            voxels: Vec4::new(TIER_VOXEL[0], TIER_VOXEL[1], TIER_VOXEL[2], 0.0),
            resolution: RESOLUTION as u32,
        }
    }
}

#[derive(ShaderType, Clone, Copy, Default)]
struct GpuLight {
    center: Vec3,
    radius: f32,
    radiance: Vec3,
}

/// Per-frame inputs, extracted to the render world.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct Cascades3dFrame {
    pub active: bool,
    pub camera: Vec3,
    pub clipmap: ClipmapUniform,
    lights: Vec<GpuLight>,
    /// Follows the clipmap's generation: a rebuild also drops cached hits.
    pub generation: u32,
}

impl Plugin for Cascades3dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Cascades3dFrame>()
            .add_plugins(ExtractResourcePlugin::<Cascades3dFrame>::default())
            .add_systems(PostUpdate, update_frame.after(crate::clipmap::ClipmapSystems))
            .add_systems(Update, log_cascade_timings);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(RenderStartup, init_cascades_gpu).add_systems(
            Core3d,
            run_cascades
                .in_set(Cascades3dRenderSystems)
                .after(ClipmapRenderSystems)
                .in_set(Core3dSystems::Prepass),
        );
    }
}

fn update_frame(
    mut frame: ResMut<Cascades3dFrame>,
    view: Option<Res<SdfView>>,
    state: Res<ClipmapState>,
    scene: Option<Res<SceneResource>>,
    camera: Query<&GlobalTransform, With<ClipmapCamera>>,
) {
    let (Some(view), Ok(camera)) = (view, camera.single()) else { return };
    frame.active = view.enabled && view.mode == CASCADE_MODE;
    frame.camera = camera.translation();
    frame.clipmap = ClipmapUniform::from_state(&state);
    frame.generation = state.generation;
    if frame.lights.is_empty()
        && let Some(scene) = scene
    {
        frame.lights = scene
            .0
            .lights
            .iter()
            .take(MAX_LIGHTS)
            .map(|l| GpuLight { center: l.center, radius: l.radius, radiance: l.radiance })
            .collect();
    }
}

fn log_cascade_timings(time: Res<Time>, mut timer: Local<Option<Timer>>, frame: Res<Cascades3dFrame>, store: Res<DiagnosticsStore>) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() || !frame.active {
        return;
    }
    let ms = |span: &str| store.get(&DiagnosticPath::new(format!("render/{span}/elapsed_gpu"))).and_then(|d| d.smoothed());
    let gathers: Vec<String> =
        (0..CASCADES).rev().filter_map(|n| ms(&gather_span(n)).map(|v| format!("c{n} {v:.2}"))).collect();
    if let Some(trace) = ms(TRACE_SPAN) {
        info!("cascades3d GPU ms: trace {trace:.2} | gather {}", gathers.join(", "));
    }
}

fn gather_span(n: u32) -> String {
    format!("cascades3d_gather_{n}")
}

// ---------------------------------------------------------------- render

#[derive(ShaderType, Clone, Copy)]
struct Params {
    clipmap: ClipmapUniform,
    camera: Vec3,
    cascade: u32,
    cascade_count: u32,
    light_count: u32,
    spacing0: f32,
    interval0: f32,
    octa_side0: u32,
    window: u32,
    radius: f32,
    tag_base: u32,
    hit_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    angle_base: u32,
    child_angle_base: u32,
}

/// GPU state shared with the SDF view, which shades from cascade 0.
#[derive(Resource)]
pub struct Cascades3dGpu {
    pub layout: Layout,
    /// Cascade 0 (even cascades write A) after the gather.
    pub radiance_a: Buffer,
    radiance_b: Buffer,
    hits: Buffer,
    pub tags: Buffer,
    /// Per slot, which trilinear children are reachable (see the shader).
    child_masks: Buffer,
    remarched: Buffer,
    solid_angles: Buffer,
    /// Per cascade-0 direction cell: sub-directions and solid angles, for
    /// cosine-weighted irradiance.
    pub shading_subsamples: Buffer,
    /// The scene's lights; written once the scene is known.
    pub lights: Buffer,
    angle_bases: Vec<u32>,
    bind_group_layout: BindGroupLayoutDescriptor,
    trace: CachedComputePipelineId,
    commit: CachedComputePipelineId,
    gather: CachedComputePipelineId,
}

fn storage(render_device: &RenderDevice, label: &str, size: u64) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn bytes_of<T: ShaderType + bevy::render::render_resource::encase::internal::WriteInto>(value: &T) -> Vec<u8> {
    let mut buffer = encase::StorageBuffer::new(Vec::new());
    buffer.write(value).unwrap();
    buffer.into_inner()
}

fn init_cascades_gpu(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = Layout::new();
    info!(
        "cascades3d: {} cascades, rays per window {:?}, hits {:.0} MB, radiance 2 x {:.0} MB",
        CASCADES,
        (0..CASCADES).map(|n| layout.cascades[n as usize].size.pow(3) * rays_per_probe(n)).collect::<Vec<_>>(),
        layout.hit_words as f64 * 4.0 / 1e6,
        layout.max_rays as f64 * 8.0 / 1e6
    );

    let mut angles = Vec::new();
    let mut angle_bases = Vec::new();
    for n in 0..CASCADES {
        angle_bases.push(angles.len() as u32);
        angles.extend(layout::octa_solid_angles(octa_side(n)));
    }
    let with_data = |label: &str, contents: &[u8]| {
        render_device.create_buffer_with_data(&BufferInitDescriptor { label: Some(label), contents, usage: BufferUsages::STORAGE })
    };
    let solid_angles = with_data("cascades3d_solid_angles", &bytes_of(&angles));
    let shading_subsamples = with_data("cascades3d_shading_subsamples", &bytes_of(&layout::shading_subsamples()));

    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "cascades3d_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Params>(true),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let shader: Handle<Shader> = asset_server.load(SHADER);
    let queue = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("cascades3d_{entry}").into()),
            layout: vec![bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };

    let (trace, commit, gather) = (queue("trace"), queue("commit"), queue("gather"));
    commands.insert_resource(Cascades3dGpu {
        radiance_a: storage(&render_device, "cascades3d_radiance_a", layout.max_rays * 8),
        radiance_b: storage(&render_device, "cascades3d_radiance_b", layout.max_rays * 8),
        hits: storage(&render_device, "cascades3d_hits", layout.hit_words * 4),
        tags: storage(&render_device, "cascades3d_tags", layout.tag_words * 4),
        child_masks: storage(&render_device, "cascades3d_child_masks", layout.tag_words * 4),
        remarched: storage(&render_device, "cascades3d_remarched", CASCADES as u64 * 4),
        lights: storage(&render_device, "cascades3d_lights", (MAX_LIGHTS * 32) as u64),
        solid_angles,
        shading_subsamples,
        angle_bases,
        bind_group_layout,
        trace,
        commit,
        gather,
        layout,
    });
}

/// Dispatches `threads` threads in workgroups of 64, spread over y when
/// they'd exceed the 65535-workgroup limit per dimension; the shader
/// recovers the linear index with `linear_index`.
fn dispatch_linear(pass: &mut ComputePass, threads: u32) {
    const MAX_X: u32 = 32768;
    let groups = threads.div_ceil(64);
    if groups <= MAX_X {
        pass.dispatch_workgroups(groups, 1, 1);
    } else {
        pass.dispatch_workgroups(MAX_X, groups.div_ceil(MAX_X), 1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cascades(
    frame: Res<Cascades3dFrame>,
    gpu: Option<Res<Cascades3dGpu>>,
    textures: Option<Res<ClipmapTextures>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: Local<DynamicUniformBuffer<Params>>,
    mut cleared_generation: Local<Option<u32>>,
    mut uploaded_lights: Local<usize>,
    mut ctx: RenderContext,
) {
    let (Some(gpu), Some(textures)) = (gpu, textures) else { return };
    if frame.lights.is_empty() {
        return;
    }
    // The SDF view's direct-light mode reads the lights too, so upload them
    // regardless of whether the cascades run.
    if *uploaded_lights != frame.lights.len() {
        render_queue.write_buffer(&gpu.lights, 0, &bytes_of(&frame.lights));
        *uploaded_lights = frame.lights.len();
    }
    if !frame.active {
        return;
    }
    for id in [gpu.trace, gpu.commit, gpu.gather] {
        if let CachedPipelineState::Err(e) = pipeline_cache.get_compute_pipeline_state(id) {
            error_once!("cascades3d pipeline failed: {e}");
        }
    }
    let (Some(trace), Some(commit), Some(gather)) = (
        pipeline_cache.get_compute_pipeline(gpu.trace),
        pipeline_cache.get_compute_pipeline(gpu.commit),
        pipeline_cache.get_compute_pipeline(gpu.gather),
    ) else {
        return;
    };
    let windows = &gpu.layout.cascades;
    params.clear();
    let offsets: Vec<u32> = (0..CASCADES)
        .map(|n| {
            let own = windows[n as usize];
            let child = windows.get(n as usize + 1).copied().unwrap_or_default();
            params.push(&Params {
                clipmap: frame.clipmap,
                camera: frame.camera,
                cascade: n,
                cascade_count: CASCADES,
                light_count: frame.lights.len() as u32,
                spacing0: SPACING0,
                interval0: INTERVAL0,
                octa_side0: OCTA_SIDE0,
                window: own.size,
                radius: own.radius,
                tag_base: own.tag_base,
                hit_base: own.hit_base,
                child_window: child.size,
                child_radius: child.radius,
                child_tag_base: child.tag_base,
                angle_base: gpu.angle_bases[n as usize],
                child_angle_base: gpu.angle_bases.get(n as usize + 1).copied().unwrap_or(0),
            })
        })
        .collect();
    params.write_buffer(&render_device, &render_queue);
    let Some(params_binding) = params.binding() else { return };

    let layout = pipeline_cache.get_bind_group_layout(&gpu.bind_group_layout);
    let bind_group = |src: &Buffer, dst: &Buffer| {
        render_device.create_bind_group(
            "cascades3d",
            &layout,
            &BindGroupEntries::sequential((
                params_binding.clone(),
                &textures.views[0],
                &textures.views[1],
                &textures.views[2],
                &textures.sampler,
                gpu.lights.as_entire_buffer_binding(),
                src.as_entire_buffer_binding(),
                dst.as_entire_buffer_binding(),
                gpu.hits.as_entire_buffer_binding(),
                gpu.tags.as_entire_buffer_binding(),
                gpu.remarched.as_entire_buffer_binding(),
                gpu.solid_angles.as_entire_buffer_binding(),
                gpu.child_masks.as_entire_buffer_binding(),
            )),
        )
    };
    let writes_a = bind_group(&gpu.radiance_b, &gpu.radiance_a);
    let writes_b = bind_group(&gpu.radiance_a, &gpu.radiance_b);

    if *cleared_generation != Some(frame.generation) {
        ctx.command_encoder().clear_buffer(&gpu.tags, 0, None);
        *cleared_generation = Some(frame.generation);
    }
    ctx.command_encoder().clear_buffer(&gpu.remarched, 0, None);

    let slots = |n: u32| windows[n as usize].size.pow(3);
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
        label: Some("cascades3d"),
        timestamp_writes: None,
    });

    let span = diagnostics.pass_span(&mut pass, TRACE_SPAN);
    pass.set_pipeline(trace);
    for n in 0..CASCADES {
        pass.set_bind_group(0, &writes_a, &[offsets[n as usize]]);
        dispatch_linear(&mut pass, slots(n) * rays_per_probe(n));
    }
    pass.set_pipeline(commit);
    for n in 0..CASCADES {
        pass.set_bind_group(0, &writes_a, &[offsets[n as usize]]);
        dispatch_linear(&mut pass, slots(n));
    }
    span.end(&mut pass);

    pass.set_pipeline(gather);
    for n in (0..CASCADES).rev() {
        let span = diagnostics.pass_span(&mut pass, gather_span(n));
        let group = if n % 2 == 0 { &writes_a } else { &writes_b };
        pass.set_bind_group(0, group, &[offsets[n as usize]]);
        dispatch_linear(&mut pass, slots(n) * rays_per_probe(n));
        span.end(&mut pass);
    }
}
