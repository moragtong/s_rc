//! 3D radiance cascades over the SDF clipmap, in world space, in
//! camera-centred windows (see layout.rs for the parameters and
//! cascades3d.wgsl for the passes). They carry indirect light: hits see the
//! clipmap's surface color times its sun visibility.
//!
//! They run while the SDF view shows lighting (`N` to modes 3 and 4); `K`
//! rebuilds the clipmap and also throws away the cached hits.

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

use crate::clipmap::{ClipmapCamera, ClipmapRenderSystems, ClipmapState, ClipmapTextures, ClipmapUniform};
use crate::sdf_view::SdfView;
use layout::{CASCADES, CascadeWindow, LEVELS, Layout, octa_side};

const SHADER: &str = "shaders/cascades3d.wgsl";
const TRACE_SPAN: &str = "cascades3d_trace";

/// First SDF view mode that shows lighting (exports radiance).
pub const LIGHTING_MODE: u32 = 3;

pub struct Cascades3dPlugin;

/// Render-world ordering: the cascades run after the clipmap is composed and
/// before anything that shades from them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cascades3dRenderSystems;

/// Per-frame inputs, extracted to the render world.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct Cascades3dFrame {
    pub active: bool,
    pub camera: Vec3,
    pub clipmap: ClipmapUniform,
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
    camera: Query<&GlobalTransform, With<ClipmapCamera>>,
) {
    let (Some(view), Ok(camera)) = (view, camera.single()) else { return };
    frame.active = view.enabled && view.mode >= LIGHTING_MODE;
    frame.camera = camera.translation();
    frame.clipmap = ClipmapUniform::from_state(&state);
    frame.generation = state.generation;
}

fn log_cascade_timings(time: Res<Time>, mut timer: Local<Option<Timer>>, frame: Res<Cascades3dFrame>, store: Res<DiagnosticsStore>) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() || !frame.active {
        return;
    }
    let ms = |span: &str| store.get(&DiagnosticPath::new(format!("render/{span}/elapsed_gpu"))).and_then(|d| d.smoothed());
    let gathers: Vec<String> = (0..CASCADES)
        .rev()
        .filter_map(|n| ms(&gather_span(n)).map(|v| format!("c{n} {v:.2}")))
        .chain((1..LEVELS).filter_map(|l| ms(&level_span(l)).map(|v| format!("level {l} {v:.2}"))))
        .collect();
    if let Some(trace) = ms(TRACE_SPAN) {
        info!("cascades3d GPU ms: trace {trace:.2} | gather {}", gathers.join(", "));
    }
}

fn gather_span(n: u32) -> String {
    format!("cascades3d_gather_{n}")
}

fn level_span(l: u32) -> String {
    format!("cascades3d_level_{l}")
}

// ---------------------------------------------------------------- render

#[derive(ShaderType, Clone, Copy)]
struct Params {
    clipmap: ClipmapUniform,
    camera: Vec3,
    /// Pass index: cascades 0.., then shading levels 1..
    pass: u32,
    interval: Vec2,
    spacing: f32,
    octa_side: u32,
    has_children: u32,
    child_spacing: f32,
    child_octa_side: u32,
    window: u32,
    radius: f32,
    tag_base: u32,
    hit_base: u32,
    /// Where this pass's rays go in the destination buffer.
    dst_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    child_angle_base: u32,
    /// DEBUG: 1 = cascade 0 without its far field (RC3D_C0_NEAR_ONLY).
    debug: u32,
}

/// GPU state shared with the SDF view, which shades from cascade 0.
#[derive(Resource)]
pub struct Cascades3dGpu {
    pub layout: Layout,
    /// Cascade 0 (even cascades write A) after the gather.
    pub radiance_a: Buffer,
    radiance_b: Buffer,
    /// A shading level's children averaged to its directions (the reduce
    /// pass), per child slot.
    reduced: Buffer,
    /// Shading levels 1.. after the gather (see layout.rs), each at its
    /// radiance_base.
    pub radiance_levels: Buffer,
    hits: Buffer,
    pub tags: Buffer,
    /// Per slot, which trilinear children are reachable (see the shader).
    child_masks: Buffer,
    remarched: Buffer,
    solid_angles: Buffer,
    /// Per cascade-0 direction cell: sub-directions and solid angles, for
    /// cosine-weighted irradiance.
    pub shading_subsamples: Buffer,
    angle_bases: Vec<u32>,
    bind_group_layout: BindGroupLayoutDescriptor,
    trace: CachedComputePipelineId,
    commit: CachedComputePipelineId,
    gather: CachedComputePipelineId,
    reduce: CachedComputePipelineId,
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
        "cascades3d: {} cascades + {} shading levels, rays per window {:?} + {:?}, hits {:.0} MB, radiance 2 x {:.0} + {:.0} MB",
        CASCADES,
        LEVELS - 1,
        layout.cascades.iter().map(CascadeWindow::rays).collect::<Vec<_>>(),
        layout.levels.iter().map(CascadeWindow::rays).collect::<Vec<_>>(),
        layout.hit_words as f64 * 4.0 / 1e6,
        layout.max_rays as f64 * 8.0 / 1e6,
        layout.level_rays as f64 * 8.0 / 1e6
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
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
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

    let (trace, commit, gather, reduce) = (queue("trace"), queue("commit"), queue("gather"), queue("reduce"));
    commands.insert_resource(Cascades3dGpu {
        radiance_a: storage(&render_device, "cascades3d_radiance_a", layout.max_rays * 8),
        radiance_b: storage(&render_device, "cascades3d_radiance_b", layout.max_rays * 8),
        reduced: storage(&render_device, "cascades3d_reduced", layout.reduced_rays.max(1) * 8),
        radiance_levels: storage(&render_device, "cascades3d_radiance_levels", layout.level_rays.max(1) * 8),
        hits: storage(&render_device, "cascades3d_hits", layout.hit_words * 4),
        tags: storage(&render_device, "cascades3d_tags", layout.tag_words * 4),
        child_masks: storage(&render_device, "cascades3d_child_masks", layout.tag_words * 4),
        remarched: storage(&render_device, "cascades3d_remarched", (CASCADES + LEVELS) as u64 * 4),
        solid_angles,
        shading_subsamples,
        angle_bases,
        bind_group_layout,
        trace,
        commit,
        gather,
        reduce,
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
    mut ctx: RenderContext,
) {
    let (Some(gpu), Some(textures)) = (gpu, textures) else { return };
    if !frame.active {
        return;
    }
    for id in [gpu.trace, gpu.commit, gpu.gather, gpu.reduce] {
        if let CachedPipelineState::Err(e) = pipeline_cache.get_compute_pipeline_state(id) {
            error_once!("cascades3d pipeline failed: {e}");
        }
    }
    let (Some(trace), Some(commit), Some(gather), Some(reduce)) = (
        pipeline_cache.get_compute_pipeline(gpu.trace),
        pipeline_cache.get_compute_pipeline(gpu.commit),
        pipeline_cache.get_compute_pipeline(gpu.gather),
        pipeline_cache.get_compute_pipeline(gpu.reduce),
    ) else {
        return;
    };
    // Passes: cascades 0.., then shading levels 1..
    let windows = &gpu.layout.cascades;
    let passes: Vec<CascadeWindow> = windows.iter().chain(&gpu.layout.levels).copied().collect();
    params.clear();
    let make = |i: usize, own: &CascadeWindow| {
        let child = own.child.map(|c| windows[c as usize]).unwrap_or_default();
        Params {
            clipmap: frame.clipmap,
            camera: frame.camera,
            pass: i as u32,
            interval: Vec2::from(own.interval),
            spacing: own.spacing,
            octa_side: own.octa_side,
            has_children: own.child.is_some() as u32,
            child_spacing: child.spacing,
            child_octa_side: child.octa_side,
            window: own.size,
            radius: own.radius,
            tag_base: own.tag_base,
            hit_base: own.hit_base,
            dst_base: own.radiance_base,
            child_window: child.size,
            child_radius: child.radius,
            child_tag_base: child.tag_base,
            child_angle_base: own.child.map(|c| gpu.angle_bases[c as usize]).unwrap_or(0),
            debug: (i == 0 && std::env::var("RC3D_C0_NEAR_ONLY").is_ok()) as u32,
        }
    };
    let mut offsets = Vec::new();
    for (i, own) in passes.iter().enumerate() {
        let mut p = make(i, own);
        if i >= CASCADES as usize {
            // Shading levels gather from the reduced children.
            p.child_octa_side = p.octa_side;
        }
        offsets.push(params.push(&p));
    }
    // The reduce passes, one per shading level, after all the passes.
    let reduce_offsets: Vec<u32> =
        gpu.layout.levels.iter().enumerate().map(|(l, own)| params.push(&make(CASCADES as usize + l, own))).collect();
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
                &textures.color[0],
                &textures.color[1],
                &textures.color[2],
                &textures.shadow[0],
                &textures.shadow[1],
                &textures.shadow[2],
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
    // Shading level l merges with cascade l+1, read from wherever that
    // cascade was just written.
    let reduce_from_a = bind_group(&gpu.radiance_a, &gpu.reduced);
    let reduce_from_b = bind_group(&gpu.radiance_b, &gpu.reduced);
    let writes_levels = bind_group(&gpu.reduced, &gpu.radiance_levels);

    // DEBUG RC3D_RETRACE_ALL: drop every cached hit each frame.
    if *cleared_generation != Some(frame.generation) || std::env::var("RC3D_RETRACE_ALL").is_ok() {
        ctx.command_encoder().clear_buffer(&gpu.tags, 0, None);
        *cleared_generation = Some(frame.generation);
    }
    ctx.command_encoder().clear_buffer(&gpu.remarched, 0, None);

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
        label: Some("cascades3d"),
        timestamp_writes: None,
    });

    let span = diagnostics.pass_span(&mut pass, TRACE_SPAN);
    pass.set_pipeline(trace);
    for (w, offset) in passes.iter().zip(&offsets) {
        pass.set_bind_group(0, &writes_a, &[*offset]);
        dispatch_linear(&mut pass, w.rays() as u32);
    }
    pass.set_pipeline(commit);
    for (w, offset) in passes.iter().zip(&offsets) {
        pass.set_bind_group(0, &writes_a, &[*offset]);
        dispatch_linear(&mut pass, w.size.pow(3));
    }
    span.end(&mut pass);

    // Top down; each shading level right after the cascade it merges with,
    // before the ping-pong overwrites that cascade.
    pass.set_pipeline(gather);
    for n in (0..CASCADES).rev() {
        let span = diagnostics.pass_span(&mut pass, gather_span(n));
        let group = if n % 2 == 0 { &writes_a } else { &writes_b };
        pass.set_bind_group(0, group, &[offsets[n as usize]]);
        dispatch_linear(&mut pass, windows[n as usize].rays() as u32);
        span.end(&mut pass);
        let level = n.wrapping_sub(1);
        if (1..LEVELS).contains(&level) {
            let span = diagnostics.pass_span(&mut pass, level_span(level));
            let index = (CASCADES + level - 1) as usize;
            let own = passes[index];
            pass.set_pipeline(reduce);
            let group = if n % 2 == 0 { &reduce_from_a } else { &reduce_from_b };
            pass.set_bind_group(0, group, &[reduce_offsets[level as usize - 1]]);
            dispatch_linear(&mut pass, windows[n as usize].size.pow(3) * own.rays_per_probe());
            pass.set_pipeline(gather);
            pass.set_bind_group(0, &writes_levels, &[offsets[index]]);
            dispatch_linear(&mut pass, own.rays() as u32);
            span.end(&mut pass);
        }
    }
}
