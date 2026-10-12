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
        texture::TextureCache,
        view::{ExtractedView, ViewTarget},
    },
};

use crate::cascades2_5d::{CascadesRenderData, common_layout};
use crate::gtao_ssgi::{GTAO_STRIDE, GtaoRenderData};
use crate::ui::LightingState;

const SHADER_PATH: &str = "shaders/shade_cascades.wgsl";

pub struct ShadePlugin;

#[derive(ShaderType, Clone)]
struct ShadeUniforms {
    view_from_clip: Mat4,
    world_from_view: Mat4,
    indirect_enabled: u32,
    trilinear_enabled: u32,
    bilateral_enabled: u32,
    normal_bias_enabled: u32,
    bsp_recon_enabled: u32,
    gtao_enabled: u32,
    gtao_stride: u32,
}

/// The composite half of the shade split: writes the indirect term (times the
/// SSAO AO when enabled) onto the HDR image.
#[derive(ShaderType, Clone)]
struct CompositeUniforms {
    ao_enabled: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
                    // The indirect term goes to its own texture now (see
                    // `draw_shade`); replace, nothing to blend with.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// The second half of the shade split: adds the indirect texel read from the
/// shade pass's own buffer back onto the lit HDR image. This is where Bevy's
/// SSAO AO is meant to multiply the indirect once it is wired in.
#[derive(Resource)]
struct CompositePipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

impl SpecializedRenderPipeline for CompositePipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: TextureFormat) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("composite_indirect".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Add the indirect light; leave alpha alone.
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
            .init_resource::<SpecializedRenderPipelines<CompositePipeline>>()
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
                    // GTAO-SSGI per-pixel irradiance; read only when enabled.
                    storage_buffer_read_only::<Vec4>(false),
                ),
            ),
        ),
        shader: asset_server.load(SHADER_PATH),
        fullscreen_shader: fullscreen_shader.clone(),
    });
    commands.insert_resource(CompositePipeline {
        layout: BindGroupLayoutDescriptor::new(
            "composite_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // The isolated indirect term, written by the shade pass.
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    // Bevy's SSAO AO (R16Float on this adapter, hence
                    // filterable). Gated off by `ao_enabled` when the SSAO
                    // component is absent; the fallback image is bound but not
                    // read then (see the shader).
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    uniform_buffer::<CompositeUniforms>(false),
                ),
            ),
        ),
        shader: asset_server.load("shaders/composite_indirect.wgsl"),
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
    bake: Res<crate::heightmap_bake::BakeState>,
    pipeline: Res<ShadePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShadePipeline>>,
    composite: Res<CompositePipeline>,
    mut composite_pipelines: ResMut<SpecializedRenderPipelines<CompositePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    fallback: Res<bevy::render::texture::FallbackImage>,
    render_data: Option<Res<CascadesRenderData>>,
    gtao: Option<Res<GtaoRenderData>>,
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
    let Some(gtao) = gtao else {
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
        trilinear_enabled: lighting_state.trilinear_enabled as u32,
        bilateral_enabled: lighting_state.bilateral_enabled as u32,
        normal_bias_enabled: lighting_state.normal_bias_enabled as u32,
        bsp_recon_enabled: lighting_state.bsp_recon_enabled as u32,
        gtao_enabled: (lighting_state.gtao_ssgi_enabled && bake.is_verified()) as u32,
        gtao_stride: GTAO_STRIDE,
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

    // Per-frame buffer for the isolated indirect term, one per view that
    // reaches this pass. `TextureCache` recycles entries frame to frame.
    let viewport = extracted_view.viewport;
    let indirect_texture = texture_cache.get(
        &render_device,
        TextureDescriptor {
            label: Some("shade_indirect_texture"),
            size: Extent3d {
                width: viewport.z,
                height: viewport.w,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: view_target.main_texture_format(),
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    let bind_group = render_device.create_bind_group(
        "shade_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &depth.texture.default_view,
            &gbuffer.texture.default_view,
            render_data.sh_probes_blend_buffer.as_entire_binding(),
            uniforms.binding().unwrap(),
            gtao.buffer.as_entire_binding(),
        )),
    );

    // The render pass's borrow of `ctx` lives until it is dropped (its Drop
    // calls `end()`), so each pass gets its own scope.
    {
        let mut pass = ctx
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("shade_pass"),
                // The indirect term lands in its own per-frame texture (so Bevy's
                // SSAO AO can multiply it before it touches the image); the
                // composite pass below adds it on. `LoadOp::Clear` is belt-and-
                // braces: the fullscreen triangle covers every pixel anyway.
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &indirect_texture.default_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })],
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

    // Composite the indirect buffer back onto the lit image (additive).
    let composite_pipeline_id = composite_pipelines.specialize(
        &pipeline_cache,
        &composite,
        view_target.main_texture_format(),
    );
    let Some(composite_pipeline) = pipeline_cache.get_render_pipeline(composite_pipeline_id) else {
        return;
    };
    let mut composite_uniforms = UniformBuffer::from(CompositeUniforms {
        ao_enabled: (ssao_resources.is_some() && lighting_state.bevy_ssao_enabled) as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    });
    composite_uniforms.write_buffer(&render_device, &render_queue);
    let composite_bind_group = render_device.create_bind_group(
        "composite_bind_group",
        &pipeline_cache.get_bind_group_layout(&composite.layout),
        &BindGroupEntries::sequential((
            &indirect_texture.default_view,
            // Bevy's AO, or any texture when the SSAO component is off: the
            // shader only reads it when `ao_enabled` is set.
            match ssao_resources {
                Some(res) => &res.screen_space_ambient_occlusion_texture.default_view,
                None => &fallback.d2.texture_view,
            },
            composite_uniforms.binding().unwrap(),
        )),
    );

    {
        let mut pass = ctx
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("composite_indirect"),
                color_attachments: &[Some(view_target.get_unsampled_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, &composite_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
