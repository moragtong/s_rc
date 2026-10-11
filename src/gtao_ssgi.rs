//! GTAO-based screen-space GI. See gtao_ssgi.wgsl.
//!
//! Runs at `Core3dSystems::EarlyPostProcess` - after deferred lighting has
//! filled the main texture and before `shade` adds indirect at `PostProcess` -
//! so the directly-lit colour it samples is this frame's own. Output is a
//! per-pixel irradiance buffer that `shade` reads when the toggle is on; the
//! world cascade probes remain the fallback for directions the screen cannot
//! answer.

use bevy::{
    core_pipeline::{Core3d, Core3dSystems, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_2d, texture_depth_2d,
                uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
    },
};

use bevy::camera::NormalizedRenderTarget;
use bevy::render::camera::ExtractedCamera;

use crate::cascades2_5d::{CascadesRenderData, common_layout};
use crate::heightmap_bake::BakeState;
use crate::ui::LightingState;

const SHADER_PATH: &str = "shaders/gtao_ssgi.wgsl";
/// Azimuth slices per pixel.
const SLICES: u32 = 6;
/// Steps of each slice's horizon march.
const STEPS: u32 = 8;
/// World radius each slice reaches, m.
const RADIUS_M: f32 = 1.5;
/// Fixed pixel-row stride of the output buffer and its height cap. A window
/// wider than `GTAO_STRIDE` or taller than `GTAO_MAX_HEIGHT` must raise these
/// (and `GTAO_STRIDE` in `shade_cascades.wgsl` mirrors it through the uniform).
pub const GTAO_STRIDE: u32 = 2048;
pub const GTAO_MAX_HEIGHT: u32 = 1152;
const GTAO_MAX_PIXELS: u64 = GTAO_STRIDE as u64 * GTAO_MAX_HEIGHT as u64;

#[derive(ShaderType, Clone)]
struct GtaoUniforms {
    clip_from_world: Mat4,
    world_from_clip: Mat4,
    eye: Vec3,
    radius_m: f32,
    screen: UVec2,
    stride: u32,
    slices: u32,
    steps: u32,
    step_bias: f32,
    enabled: u32,
    pad0: u32,
}

#[derive(Resource)]
struct GtaoPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

/// The per-pixel irradiance buffer, allocated once at a fixed size so `shade`
/// can index it without knowing the live viewport.
#[derive(Resource)]
pub struct GtaoRenderData {
    pub buffer: Buffer,
}

pub struct GtaoSsgiPlugin;

impl Plugin for GtaoSsgiPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_gtao_pipeline)
            .add_systems(Core3d, draw_gtao.in_set(Core3dSystems::EarlyPostProcess));
    }
}

fn init_gtao_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "gtao_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_depth_2d(),
                texture_2d(TextureSampleType::Uint), // deferred G-buffer
                // The view target after deferred lighting: directly-lit colour.
                texture_2d(TextureSampleType::Float { filterable: true }),
                storage_buffer_read_only::<Vec4>(false), // world sh_probes
                storage_buffer::<Vec4>(false),           // gtao output
                uniform_buffer::<GtaoUniforms>(false),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("gtao_ssgi".into()),
        layout: vec![common_layout(), layout.clone()],
        shader: asset_server.load(SHADER_PATH),
        entry_point: Some("gtao".into()),
        ..default()
    });
    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("gtao_output"),
        size: GTAO_MAX_PIXELS * 16,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    commands.insert_resource(GtaoPipeline { layout, pipeline });
    commands.insert_resource(GtaoRenderData { buffer });
}

fn draw_gtao(
    view: ViewQuery<(
        &ViewTarget,
        &ExtractedView,
        &ExtractedCamera,
        Option<&ViewPrepassTextures>,
    )>,
    lighting_state: Res<LightingState>,
    bake: Res<BakeState>,
    pipeline: Res<GtaoPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_data: Option<Res<CascadesRenderData>>,
    gtao: Option<Res<GtaoRenderData>>,
    mut ctx: RenderContext,
) {
    if !lighting_state.indirect_enabled || !lighting_state.gtao_ssgi_enabled {
        return;
    }
    if !bake.is_verified() {
        return;
    }
    let (Some(render_data), Some(gtao)) = (render_data, gtao) else {
        return;
    };
    let (view_target, extracted_view, extracted_camera, prepass) = view.into_inner();

    // Only the window camera has a screen-space pass. The startup bake cameras
    // are their own 512x512 `RenderTarget::Image` views.
    if !matches!(extracted_camera.target, Some(NormalizedRenderTarget::Window(_))) {
        return;
    }
    let Some(prepass) = prepass else {
        return;
    };
    let (Some(depth), Some(gbuffer)) = (&prepass.depth, &prepass.deferred) else {
        return;
    };

    let viewport = extracted_view.viewport;
    let screen = UVec2::new(viewport.z.max(1), viewport.w.max(1));

    let clip_from_world = extracted_view.clip_from_world.unwrap_or(
        extracted_view.clip_from_view * extracted_view.world_from_view.to_matrix().inverse(),
    );
    let world_from_clip = clip_from_world.inverse();
    let eye = extracted_view.world_from_view.translation();

    let mut uniforms = UniformBuffer::from(GtaoUniforms {
        clip_from_world,
        world_from_clip,
        eye,
        radius_m: RADIUS_M,
        screen,
        stride: GTAO_STRIDE,
        slices: SLICES,
        steps: STEPS,
        step_bias: 0.0,
        enabled: 1,
        pad0: 0,
    });
    uniforms.write_buffer(&render_device, &render_queue);

    let Some(render_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "gtao_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &depth.texture.default_view,
            &gbuffer.texture.default_view,
            view_target.main_texture_view(),
            render_data.sh_probes_blend_buffer.as_entire_binding(),
            gtao.buffer.as_entire_binding(),
            uniforms.binding().unwrap(),
        )),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("gtao_ssgi"),
            ..default()
        });
    pass.set_pipeline(render_pipeline);
    pass.set_bind_group(0, &render_data.common_bind_group, &[]);
    pass.set_bind_group(1, &bind_group, &[]);
    pass.dispatch_workgroups(screen.x.div_ceil(8), screen.y.div_ceil(8), 1);
}