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
const GATHER_SHADER_PATH: &str = "shaders/gather_cascades.wgsl";
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
    raymarch_layout: BindGroupLayoutDescriptor,
    raymarch_pipeline_id: CachedComputePipelineId,
    gather_layout: BindGroupLayoutDescriptor,
    gather_pipeline_id: CachedComputePipelineId,
}

#[derive(bevy::render::render_resource::ShaderType, Clone, Copy)]
pub struct CascadesUniforms {
    hmap_min: Vec2,
    hmap_max: Vec2,
    hmap_res: UVec2,
    spacing: f32,
    octa_side: u32,
    grid_size: UVec2,
}

#[derive(Resource)]
struct CascadesRenderData {
    hit_buffer: Buffer,
    sh_probes_buffer: Buffer,
    uniform_buffer: Buffer,
    raymarch_bind_group: BindGroup,
    gather_bind_group: BindGroup,
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
                dispatch_raymarch.in_set(Core3dSystems::Prepass),
            );
    }
}

fn setup_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let raymarch_layout = BindGroupLayoutDescriptor::new(
        "cascades_raymarch_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer::<f32>(false),
                uniform_buffer::<CascadesUniforms>(false),
            ),
        ),
    );

    let raymarch_shader = asset_server.load(SHADER_PATH);
    let raymarch_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cascades2_5d_raymarch".into()),
        layout: vec![raymarch_layout.clone()],
        shader: raymarch_shader,
        shader_defs: vec![],
        entry_point: Some("cascade0".into()),
        ..default()
    });

    let gather_layout = BindGroupLayoutDescriptor::new(
        "cascades_gather_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // heightmap
                texture_2d(TextureSampleType::Float { filterable: false }), // metadata_min
                texture_2d(TextureSampleType::Float { filterable: false }), // metadata_max
                storage_buffer::<f32>(false), // hits
                storage_buffer::<Vec4>(false), // sh_probes
                uniform_buffer::<CascadesUniforms>(false),
            ),
        ),
    );

    let gather_shader = asset_server.load(GATHER_SHADER_PATH);
    let gather_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cascades2_5d_gather".into()),
        layout: vec![gather_layout.clone()],
        shader: gather_shader,
        shader_defs: vec![],
        entry_point: Some("gather".into()),
        ..default()
    });

    commands.insert_resource(CascadesPipeline { 
        raymarch_layout, 
        raymarch_pipeline_id,
        gather_layout,
        gather_pipeline_id,
    });
}

fn prepare_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    layout: Res<CascadesLayout>,
    heightmap: Option<Res<Heightmap>>,
    existing_data: Option<Res<CascadesRenderData>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(heightmap) = heightmap else { return; };
    let Some(image) = gpu_images.get(&heightmap.texture) else { return; };
    let Some(meta_min_img) = gpu_images.get(&heightmap.metadata_min) else { return; };
    let Some(meta_max_img) = gpu_images.get(&heightmap.metadata_max) else { return; };

    let c0 = &layout.layout.cascades[0];
    let grid_size = c0.size;

    if existing_data.is_none() {
        let hit_buffer_size = (layout.layout.hit_words * 4) as u64;
        let num_nodes = (c0.size.x * c0.size.y) as u64;
        let sh_buffer_size = num_nodes * 9 * 16; // 9 vec4s

        let hit_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("cascades_probes_buffer"),
            size: hit_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sh_probes_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("cascades_sh_probes_buffer"),
            size: sh_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = CascadesUniforms {
            hmap_min: heightmap.min_bounds,
            hmap_max: heightmap.max_bounds,
            hmap_res: heightmap.resolution,
            spacing: layout.layout.cascades[0].spacing,
            octa_side: layout.layout.cascades[0].octa_side,
            grid_size,
        };

        let mut uniform_buffer = bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
        uniform_buffer.write(&uniforms).unwrap();
        
        let u_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("cascades_uniforms"),
            contents: uniform_buffer.as_ref(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let rm_layout = pipeline_cache.get_bind_group_layout(&pipeline.raymarch_layout);
        let raymarch_bind_group = render_device.create_bind_group(
            "cascades_raymarch_bind_group",
            &rm_layout,
            &BindGroupEntries::sequential((
                &image.texture_view,
                hit_buffer.as_entire_binding(),
                u_buffer.as_entire_binding(),
            )),
        );

        let gth_layout = pipeline_cache.get_bind_group_layout(&pipeline.gather_layout);
        let gather_bind_group = render_device.create_bind_group(
            "cascades_gather_bind_group",
            &gth_layout,
            &BindGroupEntries::sequential((
                &image.texture_view, // heightmap needed for origin reconstruction
                &meta_min_img.texture_view,
                &meta_max_img.texture_view,
                hit_buffer.as_entire_binding(),
                sh_probes_buffer.as_entire_binding(),
                u_buffer.as_entire_binding(),
            )),
        );

        commands.insert_resource(CascadesRenderData {
            hit_buffer,
            sh_probes_buffer,
            uniform_buffer: u_buffer,
            raymarch_bind_group,
            gather_bind_group,
        });
    }
}

fn dispatch_raymarch(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    render_data: Option<Res<CascadesRenderData>>,
    layout: Res<CascadesLayout>,
    mut ctx: RenderContext,
) {
    let Some(render_data) = render_data else { return; };
    let Some(raymarch_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.raymarch_pipeline_id) else {
        return;
    };
    let Some(gather_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.gather_pipeline_id) else {
        return;
    };

    let cascades0 = &layout.layout.cascades[0];
    let grid_size = cascades0.size;

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("cascades2_5d_compute"),
            timestamp_writes: None,
        });

    pass.set_pipeline(raymarch_pipeline);
    pass.set_bind_group(0, &render_data.raymarch_bind_group, &[]);
    pass.dispatch_workgroups(
        grid_size.x.div_ceil(WORKGROUP_SIZE),
        grid_size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
    
    pass.set_pipeline(gather_pipeline);
    pass.set_bind_group(0, &render_data.gather_bind_group, &[]);
    pass.dispatch_workgroups(
        grid_size.x.div_ceil(WORKGROUP_SIZE),
        grid_size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
}
