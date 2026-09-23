//! Fullscreen visualization of one Hi-Z mip, for eyeballing the pyramid.
//!
//! Controls: `H` toggles, `[` / `]` step the mip level.
//! See `assets/shaders/hi_z_debug.wgsl` for what the colors mean.

use bevy::{
    core_pipeline::{
        Core3d, Core3dSystems, FullscreenShader, tonemapping::tonemapping,
    },
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
    },
};

use crate::hi_z::ViewHiZ;

const SHADER_PATH: &str = "shaders/hi_z_debug.wgsl";
/// Distance that maps to black in the gray ramp.
const MAX_DISTANCE: f32 = 20.0;

pub struct HiZDebugPlugin;

#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct HiZDebug {
    pub enabled: bool,
    pub mip: u32,
}

#[derive(ShaderType, Clone, Copy)]
struct HiZDebugUniform {
    mip: u32,
    near: f32,
    max_distance: f32,
}

#[derive(Resource)]
struct HiZDebugPipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

impl SpecializedRenderPipeline for HiZDebugPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: TextureFormat) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("hi_z_debug".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

impl Plugin for HiZDebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HiZDebug>()
            .add_plugins(ExtractResourcePlugin::<HiZDebug>::default())
            .add_systems(Update, hi_z_debug_input);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<HiZDebugPipeline>>()
            .add_systems(RenderStartup, init_hi_z_debug_pipeline)
            // After tonemapping so the gray ramp isn't tonemapped, and
            // before upscaling so it lands on screen.
            .add_systems(
                Core3d,
                draw_hi_z_debug
                    .after(tonemapping)
                    .in_set(Core3dSystems::PostProcess),
            );
    }
}

fn hi_z_debug_input(keys: Res<ButtonInput<KeyCode>>, mut debug: ResMut<HiZDebug>) {
    let before = *debug;
    if keys.just_pressed(KeyCode::KeyH) {
        debug.enabled = !debug.enabled;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        debug.mip += 1;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        debug.mip = debug.mip.saturating_sub(1);
    }
    if debug.enabled != before.enabled || debug.mip != before.mip {
        // The shader clamps to the pyramid's last level, so a mip past the
        // end just keeps showing the 1x1 top.
        let HiZDebug { enabled, mip } = *debug;
        info!("Hi-Z debug: enabled={enabled} mip={mip}");
    }
}

fn init_hi_z_debug_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    commands.insert_resource(HiZDebugPipeline {
        layout: BindGroupLayoutDescriptor::new(
            "hi_z_debug_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<HiZDebugUniform>(false),
                ),
            ),
        ),
        shader: asset_server.load(SHADER_PATH),
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

// Debug-only, so the pipeline lookup and per-frame uniform/bind group live
// here instead of in separate prepare systems.
fn draw_hi_z_debug(
    view: ViewQuery<(&ViewTarget, &ViewHiZ, &ExtractedView)>,
    debug: Res<HiZDebug>,
    pipeline: Res<HiZDebugPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<HiZDebugPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    if !debug.enabled {
        return;
    }
    let (view_target, hi_z, extracted_view) = view.into_inner();

    let pipeline_id =
        pipelines.specialize(&pipeline_cache, &pipeline, view_target.main_texture_format());
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };

    let mut uniform = UniformBuffer::from(HiZDebugUniform {
        mip: debug.mip,
        // Perspective infinite reverse-Z keeps `near` at clip_from_view[3][2].
        near: extracted_view.clip_from_view.w_axis.z,
        max_distance: MAX_DISTANCE,
    });
    uniform.write_buffer(&render_device, &render_queue);

    let bind_group = render_device.create_bind_group(
        "hi_z_debug",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((&hi_z.texture.default_view, uniform.binding().unwrap())),
    );

    let mut pass = ctx.command_encoder().begin_render_pass(&RenderPassDescriptor {
        label: Some("hi_z_debug"),
        color_attachments: &[Some(view_target.get_unsampled_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}
