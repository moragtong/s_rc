//! Adds indirect diffuse from the cascades' SH probes onto the lit HDR
//! image, before tonemapping. See shade_cascades.wgsl.

use bevy::{
    core_pipeline::{
        Core3d, Core3dSystems, FullscreenShader, prepass::ViewPrepassTextures,
        tonemapping::tonemapping,
    },
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        render_resource::{
            binding_types::{
                storage_buffer_read_only, texture_2d, texture_depth_2d, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
    },
};

use crate::cascades2_5d::{CascadesRenderData, common_layout};
use crate::ui::LightingState;

const SHADER_PATH: &str = "shaders/shade_cascades.wgsl";

pub struct ShadePlugin;

#[derive(ShaderType, Clone)]
struct ShadeUniforms {
    view_from_clip: Mat4,
    world_from_view: Mat4,
    indirect_enabled: u32,
    normal_bias_enabled: u32,
    use_vis_mask: u32,
    ao_enabled: u32,
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
            layout: vec![common_layout(), self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Add the indirect diffuse (times AO) directly onto the lit HDR image; leave alpha alone.
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
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
                draw_shade
                    .in_set(Core3dSystems::PostProcess)
                    .before(tonemapping),
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
                    texture_depth_2d(),
                    texture_2d(TextureSampleType::Uint), // deferred G-buffer
                    // The blended probe set; `blend_sh` writes it every frame.
                    storage_buffer_read_only::<Vec4>(false),
                    uniform_buffer::<ShadeUniforms>(false),
                    // Bevy's SSAO AO (R16Float, filterable). Gated off by `ao_enabled`.
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
            ),
        ),
        shader: asset_server.load(SHADER_PATH),
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

fn draw_shade(
    view: ViewQuery<(
        &ViewTarget,
        &ExtractedView,
        Option<&ViewPrepassTextures>,
        Option<&bevy::pbr::ScreenSpaceAmbientOcclusionResources>,
    )>,
    lighting_state: Res<LightingState>,
    pipeline: Res<ShadePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShadePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    fallback: Res<bevy::render::texture::FallbackImage>,
    render_data: Option<Res<CascadesRenderData>>,
    mut ctx: RenderContext,
) {
    // Probe display moved out to `probe_overlay` (instanced spheres); this
    // pass is only the indirect diffuse add, so the probes toggle no longer
    // keeps it alive.
    if !lighting_state.indirect_enabled {
        return;
    }
    let Some(render_data) = render_data else {
        return;
    };
    let (view_target, extracted_view, prepass, ssao_resources) = view.into_inner();
    let Some(prepass) = prepass else {
        return;
    };
    let (Some(depth), Some(gbuffer)) = (&prepass.depth, &prepass.deferred) else {
        return;
    };

    let mut uniforms = UniformBuffer::from(ShadeUniforms {
        view_from_clip: extracted_view.clip_from_view.inverse(),
        world_from_view: extracted_view.world_from_view.to_matrix(),
        indirect_enabled: lighting_state.indirect_enabled as u32,
        normal_bias_enabled: lighting_state.normal_bias_enabled as u32,
        use_vis_mask: lighting_state.use_vis_mask as u32,
        ao_enabled: (ssao_resources.is_some() && lighting_state.bevy_ssao_enabled) as u32,
    });
    uniforms.write_buffer(&render_device, &render_queue);

    let pipeline_id = pipelines.specialize(
        &pipeline_cache,
        &pipeline,
        view_target.main_texture_format(),
    );
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };

    let ao_view = match ssao_resources {
        Some(res) => &res.screen_space_ambient_occlusion_texture.default_view,
        None => &fallback.d2.texture_view,
    };

    let bind_group = render_device.create_bind_group(
        "shade_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &depth.texture.default_view,
            &gbuffer.texture.default_view,
            render_data.sh_probes_blend_buffer.as_entire_binding(),
            uniforms.binding().unwrap(),
            ao_view,
        )),
    );

    {
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
        pass.set_bind_group(0, &render_data.common_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
