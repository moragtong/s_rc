use bevy::{
    core_pipeline::{prepass::ViewPrepassTextures, Core3d, Core3dSystems, FullscreenShader},
    prelude::*,
    render::{
        render_resource::{
            binding_types::{storage_buffer, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
        RenderApp, RenderStartup,
    },
};

use crate::cascades2_5d::{CascadesLayout, CascadesRenderData, CascadesUniforms};
use crate::ui::LightingState;

const SHADER_PATH: &str = "shaders/shade_cascades.wgsl";

pub struct ShadePlugin;

#[derive(Resource)]
#[derive(ShaderType, Clone)]
struct ShadeUniforms {
    clip_from_view: Mat4,
    world_from_view: Mat4,
}

#[derive(Resource)]
struct ShadePipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

impl SpecializedRenderPipeline for ShadePipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: TextureFormat) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("shade_cascades".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState { color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add }, alpha: BlendComponent::REPLACE }), // Additive blending for indirect light
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

impl Plugin for ShadePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<ShadePipeline>>()
            .add_systems(RenderStartup, init_shade_pipeline)
            .add_systems(
                Core3d,
                draw_shade.in_set(Core3dSystems::PostProcess), // additive over tonemapped maybe, or just before tonemapping? Let's add it before tonemapping if possible. But for mockup, after might be fine. We'll set add blend.
            );
    }
}

fn init_shade_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    commands.insert_resource(ShadePipeline {
        layout: BindGroupLayoutDescriptor::new(
            "shade_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Depth), // Depth
                    texture_2d(TextureSampleType::Float { filterable: false }), // Normal
                    storage_buffer::<Vec4>(false),        // sh_probes
                    uniform_buffer::<CascadesUniforms>(false),
                    uniform_buffer::<ShadeUniforms>(false),
                ),
            ),
        ),
        shader: asset_server.load(SHADER_PATH),
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

fn draw_shade(
    view: ViewQuery<(&ViewTarget, &ExtractedView, Option<&ViewPrepassTextures>)>,
    lighting_state: Res<crate::ui::LightingState>,
    pipeline: Res<ShadePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShadePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_data: Option<Res<CascadesRenderData>>,
    mut ctx: RenderContext,
) {
    if !lighting_state.indirect_enabled {
        return;
    }

    let Some(render_data) = render_data else {
        return;
    };
    let (view_target, extracted_view, prepass) = view.into_inner();
    let Some(prepass) = prepass else {
        return;
    };

    let Some(depth_tex) = &prepass.depth else {
        return;
    };
    let Some(normal_tex) = &prepass.normal else {
        return;
    };
    let mut uni = UniformBuffer::from(ShadeUniforms {
        clip_from_view: extracted_view.clip_from_view,
        world_from_view: extracted_view.world_from_view.compute_transform().to_matrix(),
    });
    uni.write_buffer(&render_device, &render_queue);


    let pipeline_id = pipelines.specialize(
        &pipeline_cache,
        &pipeline,
        view_target.main_texture_format(),
    );
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "shade_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &depth_tex.texture.default_view,
            &normal_tex.texture.default_view,
            render_data.sh_probes_buffer.as_entire_binding(),
            render_data.uniform_buffer.as_entire_binding(),
            uni.binding().unwrap(),
        )),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("shade_pass"),
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
