pub mod layout;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        RenderApp, Render, RenderSystems,
        render_resource::{
            *,
            binding_types::{storage_buffer, texture_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        texture::GpuImage,
    },
};
use bevy::render::render_asset::RenderAssets;
use crate::heightmap::Heightmap;

const SHADER_PATH: &str = "shaders/cascades2_5d.wgsl";
const WORKGROUP_SIZE: u32 = 8; // 8x8 in compute shader

pub struct Cascades2_5dPlugin;

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

// Stores the pipeline and the global layout.
#[derive(Resource)]
struct CascadesPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
}


#[derive(bevy::render::render_resource::ShaderType, Clone, Copy)]
pub struct CascadesUniforms {
    hmap_min: Vec2,
    hmap_max: Vec2,
    hmap_res: UVec2,
    spacing: f32,
    octa_side: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Resource)]
struct CascadesRenderData {
    probes_buffer: Buffer,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
}


impl Plugin for Cascades2_5dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CascadesLayout>();
        app.add_plugins(ExtractResourcePlugin::<CascadesLayout>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(bevy::render::RenderStartup, setup_pipelines)
            .add_systems(
                Render,
                prepare_buffers.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Core3d,
                dispatch_cascade0.in_set(Core3dSystems::Prepass),
            );
    }
}

fn setup_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "cascades2_5d_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer::<f32>(false),
                uniform_buffer::<CascadesUniforms>(false),
            ),
        ),
    );

    let shader = asset_server.load(SHADER_PATH);
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cascades2_5d_cascade0".into()),
        layout: vec![layout.clone()],
        shader,
        shader_defs: vec![],
        entry_point: Some("cascade0".into()),
        ..default()
    });

    commands.insert_resource(CascadesPipeline { layout, pipeline_id });
}

fn prepare_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    layout: Res<CascadesLayout>,
    heightmap: Option<Res<Heightmap>>,
    mut existing_data: Option<ResMut<CascadesRenderData>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(heightmap) = heightmap else { return; };
    let Some(image) = gpu_images.get(&heightmap.texture) else { return; };

    if existing_data.is_none() {
        let buffer_size = (layout.layout.hit_words * 4) as u64;
        let probes_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("cascades_probes_buffer"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = CascadesUniforms {
            hmap_min: heightmap.min_bounds,
            hmap_max: heightmap.max_bounds,
            hmap_res: heightmap.resolution,
            spacing: layout.layout.cascades[0].spacing,
            octa_side: layout.layout.cascades[0].octa_side,
            _pad1: 0,
            _pad2: 0,
        };

        let mut uniform_buffer = bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
        uniform_buffer.write(&uniforms).unwrap();
        
        let u_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("cascades_uniforms"),
            contents: uniform_buffer.as_ref(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group_layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
        let bind_group = render_device.create_bind_group(
            "cascades_bind_group",
            &bind_group_layout,
            &BindGroupEntries::sequential((
                &image.texture_view,
                probes_buffer.as_entire_binding(),
                u_buffer.as_entire_binding(),
            )),
        );

        commands.insert_resource(CascadesRenderData {
            probes_buffer,
            uniform_buffer: u_buffer,
            bind_group,
        });
    }
}

fn dispatch_cascade0(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    render_data: Option<Res<CascadesRenderData>>,
    layout: Res<CascadesLayout>,
    mut ctx: RenderContext,
) {
    let Some(render_data) = render_data else { return; };
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
        return;
    };

    let cascades0 = &layout.layout.cascades[0];
    let grid_size = cascades0.size;

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("cascades2_5d_cascade0"),
            timestamp_writes: None,
        });

    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, &render_data.bind_group, &[]);
    
    pass.dispatch_workgroups(
        grid_size.x.div_ceil(WORKGROUP_SIZE),
        grid_size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
}
