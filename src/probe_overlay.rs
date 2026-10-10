//! Instanced probe-sphere debug overlay. Rasterizes a unit icosphere once per
//! probe-with-layer across all cascades in a single draw call, reading each
//! probe's baked world position straight from `probe_positions` (the same
//! buffer the trace/gather/shade passes use), so the spheres sit at the real
//! two-layer positions - no ray tracing in the fragment shader, no CPU
//! replication of the bake. Each sphere is shaded by the SH probe it stands
//! for, read from `sh_probes_blend_buffer` (the same cross-faded buffer the
//! shade pass samples), so it shows the light its probe captured rather than a
//! flat per-cascade tint. It draws into the window camera's own framebuffer
//! and tests against that view's prepass depth, so the spheres are occluded by
//! geometry exactly as the scene is. Toggled by P / `S_RC_PROBES=1`.
//!
//! See probe_overlay.wgsl. The per-cascade ranges needed to map an instance
//! index back to its cascade (and radius = spacing * scale) come from the
//! CPU-side [`CascadesLayout`], which shares its buffer layout with the GPU.

use bevy::{
    core_pipeline::{Core3d, Core3dSystems, prepass::ViewPrepassTextures, tonemapping::tonemapping},
    math::primitives::Sphere,
    mesh::{Indices, Meshable, VertexAttributeValues, VertexBufferLayout},
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        render_resource::{
            binding_types::{storage_buffer_read_only, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
    },
};

use bevy::camera::NormalizedRenderTarget;
use bevy::render::camera::ExtractedCamera;

use crate::cascades2_5d::layout::{self, CASCADES};
use crate::cascades2_5d::{CascadesLayout, CascadesRenderData};
use crate::heightmap_bake::BakeState;
use crate::ui::LightingState;

const SHADER_PATH: &str = "shaders/probe_overlay.wgsl";

/// Multiplier on the cascade spacing that sizes the drawn spheres. The probe
/// overlay's whole job is to be *seen*: the old shade-pass spheres traced at
/// 0.4x and still covered 1.8% of a wall-camera frame. Tuned down from 0.3 once
/// the sphere index count was fixed and the meshes became real spheres instead
/// of caps - at 0.3 the full spheres (and the four cascades stacking on the same
/// surface) covered ~80% of the frame and read as a solid mass.
const PROBE_RADIUS_SCALE: f32 = 0.12;

#[derive(ShaderType, Clone, Copy, Default)]
struct ProbeOverlayUniforms {
    clip_from_world: Mat4,
    /// One vec4 per cascade: (first probe-with-layer index, probe count,
    /// sphere radius, pad).
    params: [Vec4; CASCADES as usize],
}

#[derive(Resource)]
struct ProbeOverlayPipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

/// The unit icosphere's vertex and index buffers, built once at startup.
#[derive(Resource)]
struct ProbeOverlayBuffers {
    vertex: Buffer,
    index: Buffer,
    index_len: u32,
}

impl SpecializedRenderPipeline for ProbeOverlayPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: TextureFormat) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("probe_overlay".into()),
            layout: vec![self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some("vs_main".into()),
                buffers: vec![VertexBufferLayout::from_vertex_formats(
                    VertexStepMode::Vertex,
                    [VertexFormat::Float32x3],
                )],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some("fs_main".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Opaque: each sphere shows its own probe's SH, so it is
                    // the subject rather than a wash over the lit scene. The
                    // old additive tint blended neighbouring spheres together
                    // and read as a single layer.
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // Depth-test against the scene's own prepass depth (reverse-Z, so
            // `GreaterEqual` passes the closer fragment), and *write* it so the
            // spheres also occlude one another. Writing is what the previous
            // `Some(false)` broke: without it every sphere fragment at a pixel
            // passed against the scene depth and they overwrote each other in
            // instance order, so overlapping spheres (the four cascades stack on
            // the same surface) interleaved by draw order - the "weird blending".
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: default(),
                bias: default(),
            }),
            ..default()
        }
    }
}

pub struct ProbeOverlayPlugin;

impl Plugin for ProbeOverlayPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<ProbeOverlayPipeline>>()
            .add_systems(RenderStartup, (init_probe_overlay_pipeline, init_probe_overlay_buffers))
            .add_systems(
                Core3d,
                draw_probe_overlay
                    .in_set(Core3dSystems::PostProcess)
                    .before(tonemapping),
            );
    }
}

fn init_probe_overlay_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(ProbeOverlayPipeline {
        layout: BindGroupLayoutDescriptor::new(
            "probe_overlay_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer::<ProbeOverlayUniforms>(false),
                    storage_buffer_read_only::<Vec4>(false),
                    storage_buffer_read_only::<Vec4>(false),
                ),
            ),
        ),
        shader: asset_server.load(SHADER_PATH),
    });
}

fn init_probe_overlay_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    // Subdivision 2: 162 vertices, 320 triangles. The mesa is one draw call
    // over thousands of instances, so the budget is per instance.
    let mesh = Sphere::new(1.0)
        .mesh()
        .ico(2)
        .expect("two icosphere subdivisions is far under the 65535-vertex cap");
    let VertexAttributeValues::Float32x3(points) = mesh
        .attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION)
        .expect("icosphere has positions")
    else {
        unreachable!("icosphere positions are Float32x3");
    };
    let indices = match mesh.indices() {
        Some(Indices::U32(v)) => v,
        _ => unreachable!("icosphere indices are u32"),
    };

    let mut vertex_bytes = Vec::with_capacity(points.len() * 12);
    for point in points {
        for f in point {
            vertex_bytes.extend_from_slice(&f.to_ne_bytes());
        }
    }
    let mut index_bytes: Vec<u8> = Vec::with_capacity(indices.len() * 4);
    for i in indices {
        index_bytes.extend_from_slice(&i.to_ne_bytes());
    }

    commands.insert_resource(ProbeOverlayBuffers {
        vertex: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("probe_overlay_sphere_verts"),
            contents: &vertex_bytes,
            usage: BufferUsages::VERTEX,
        }),
        index: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("probe_overlay_sphere_indices"),
            contents: &index_bytes,
            usage: BufferUsages::INDEX,
        }),
        // `indices` is the raw `u32` index list (320 triangles -> 960 entries),
        // so its length is already the vertex-index count `draw_indexed` wants.
        // Dividing by 4 drew only the first 80 triangles - the 5 contiguous
        // faces of the icosahedron's cap - which rendered as a dome/hemisphere
        // instead of a sphere.
        index_len: indices.len() as u32,
    });
}

fn draw_probe_overlay(
    view: ViewQuery<(
        &ViewTarget,
        &ExtractedView,
        &ExtractedCamera,
        Option<&ViewPrepassTextures>,
    )>,
    bake: Res<BakeState>,
    lighting_state: Res<LightingState>,
    cascades_layout: Res<CascadesLayout>,
    pipeline: Res<ProbeOverlayPipeline>,
    buffers: Res<ProbeOverlayBuffers>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ProbeOverlayPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_data: Option<Res<CascadesRenderData>>,
    mut ctx: RenderContext,
) {
    // Nothing to draw until the bake has finished and written `probe_positions`.
    if !lighting_state.probes_enabled || !bake.is_done() {
        return;
    }
    let Some(render_data) = render_data else {
        return;
    };
    let (view_target, extracted_view, extracted_camera, prepass) = view.into_inner();

    // Only the window camera draws the grid. The two startup bake cameras are
    // live 3D views of their own (512x512 `RenderTarget::Image`, BAKE_LAYER);
    // without this guard the overlay splats spheres into their targets too.
    if !matches!(extracted_camera.target, Some(NormalizedRenderTarget::Window(_))) {
        return;
    }
    // Occlude against the scene's own depth buffer; without it the spheres
    // draw through walls.
    let Some(depth_view) = prepass.and_then(|p| p.depth_view()) else {
        return;
    };

    // Cascade ranges in the concatenated `probe_positions` buffer, in the
    // same probe-with-layer units the pass indexes with (`sh_base`).
    let mut params = [Vec4::ZERO; CASCADES as usize];
    for (p, c) in params.iter_mut().zip(&cascades_layout.layout.cascades) {
        *p = Vec4::new(
            c.sh_base as f32,
            (c.size.x * c.size.y * layout::LAYERS) as f32,
            c.spacing * PROBE_RADIUS_SCALE,
            0.0,
        );
    }
    let instance_count: u32 = params.iter().map(|p| p.y as u32).sum();

    // World -> clip. `world_from_view` is the camera's world matrix
    // (world <- view), so world -> view is its inverse.
    let view_from_world = extracted_view.world_from_view.to_matrix().inverse();
    let clip_from_world = extracted_view
        .clip_from_world
        .unwrap_or(extracted_view.clip_from_view * view_from_world);

    let mut uniforms = UniformBuffer::from(ProbeOverlayUniforms {
        clip_from_world,
        params,
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

    let bind_group = render_device.create_bind_group(
        "probe_overlay_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            uniforms.binding().unwrap(),
            render_data.probe_positions_buffer.as_entire_binding(),
            render_data.sh_probes_blend_buffer.as_entire_binding(),
        )),
    );

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("probe_overlay_pass"),
        color_attachments: &[Some(view_target.get_unsampled_color_attachment())],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_vertex_buffer(0, buffers.vertex.slice(..));
    pass.set_index_buffer(buffers.index.slice(..), IndexFormat::Uint32);
    pass.draw_indexed(0..buffers.index_len, 0, 0..instance_count);
}