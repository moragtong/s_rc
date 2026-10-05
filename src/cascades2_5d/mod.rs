//! 2.5D radiance cascades over the dual heightmap. Per frame: light the
//! heightmap's floor and ceiling surfaces (sun, shadowed through the
//! heightmap), trace and merge the cascades top-down with the bilinear fix,
//! then project every probe onto L2 SH with its parallax radius for
//! `shade`.

pub mod layout;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_2d, texture_storage_2d,
                uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
};

use crate::heightmap::Heightmap;

const TRACE_LIBRARY_PATH: &str = "shaders/heightmap_trace.wgsl";
const LIGHT_SHADER_PATH: &str = "shaders/surface_light.wgsl";
const TRACE_SHADER_PATH: &str = "shaders/cascades2_5d.wgsl";
const GATHER_SHADER_PATH: &str = "shaders/gather_cascades.wgsl";
/// Workgroup sizes: 8x8 texels for the light pass, 64 rays or probes for
/// the others.
const LIGHT_WORKGROUP: u32 = 8;
const WORKGROUP: u32 = 64;
/// SH probe stride in vec4s: 9 coefficients, then (position, radius).
pub const SH_STRIDE: u64 = 10;

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

/// The scene's directional light, in the units the cascades work in.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct SunParams {
    /// Unit vector toward the sun.
    pub toward: Vec3,
    /// Irradiance times the camera exposure, so cascade radiance matches the
    /// main pass's output.
    pub irradiance: Vec3,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub struct CascadeInfoUniform {
    pub size: UVec2,
    pub interval: Vec2,
    pub ray_base: u32,
    pub probe_base: u32,
    pub spacing: f32,
    pub octa_side: u32,
    pub sub: u32,
    pub sub_base: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub struct CascadesUniforms {
    pub hmap_min: Vec2,
    pub hmap_max: Vec2,
    pub hmap_res: UVec2,
    pub cascade_count: u32,
    pub hmap_top: f32,
    pub sun_toward: Vec3,
    pub sun_irradiance: Vec3,
    pub cascades: [CascadeInfoUniform; layout::CASCADES as usize],
}

#[derive(Resource)]
struct CascadesPipeline {
    /// Keeps the shared WGSL module loaded so the passes can import it.
    _trace_library: Handle<Shader>,
    light_layout: BindGroupLayoutDescriptor,
    trace_layout: BindGroupLayoutDescriptor,
    gather_layout: BindGroupLayoutDescriptor,
    light_pipeline: CachedComputePipelineId,
    trace_pipelines: Vec<CachedComputePipelineId>,
    gather_pipelines: Vec<CachedComputePipelineId>,
}

#[derive(Resource)]
pub struct CascadesRenderData {
    pub rays_buffer: Buffer,
    pub sh_probes_buffer: Buffer,
    subsamples_buffer: Buffer,
    /// Group 0 of every pass: the heightmap and the uniforms.
    pub common_bind_group: BindGroup,
    light_bind_group: BindGroup,
    trace_bind_group: BindGroup,
    gather_bind_group: BindGroup,
}

/// Group 0, shared with the shading pass.
pub fn common_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "cascades_common_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // heightmap
                uniform_buffer::<CascadesUniforms>(false),
                // Ceiling and floor surface normals. Probe relocation pushes a
                // probe out of the surface it is buried behind, along that
                // surface's normal, so the trace and gather passes need them
                // alongside the heights they came from.
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    )
}

impl Plugin for Cascades2_5dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CascadesLayout>()
            .init_resource::<SunParams>()
            .add_plugins((
                ExtractResourcePlugin::<CascadesLayout>::default(),
                ExtractResourcePlugin::<SunParams>::default(),
            ))
            .add_systems(PostUpdate, update_sun);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(bevy::render::RenderStartup, setup_pipelines)
            .add_systems(
                Render,
                prepare_buffers.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Core3d, dispatch_cascades.in_set(Core3dSystems::Prepass));
    }
}

fn update_sun(
    lights: Query<(&DirectionalLight, &GlobalTransform)>,
    cameras: Query<Option<&bevy::camera::Exposure>, With<Camera3d>>,
    mut sun: ResMut<SunParams>,
) {
    let Some((light, transform)) = lights.iter().next() else {
        return;
    };
    let exposure = cameras
        .iter()
        .next()
        .flatten()
        .copied()
        .unwrap_or_default()
        .exposure();
    let color = light.color.to_linear();
    let new = SunParams {
        toward: -transform.forward().as_vec3(),
        irradiance: Vec3::new(color.red, color.green, color.blue) * light.illuminance * exposure,
    };
    if new.toward != sun.toward || new.irradiance != sun.irradiance {
        *sun = new;
    }
}

fn setup_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let light_layout = BindGroupLayoutDescriptor::new(
        "cascades_light_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // normal_min
                texture_2d(TextureSampleType::Float { filterable: false }), // normal_max
                texture_2d(TextureSampleType::Float { filterable: false }), // albedo_min
                texture_2d(TextureSampleType::Float { filterable: false }), // albedo_max
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let trace_layout = BindGroupLayoutDescriptor::new(
        "cascades_trace_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // radiance_min
                texture_2d(TextureSampleType::Float { filterable: false }), // radiance_max
                storage_buffer::<Vec4>(false),                              // rays
            ),
        ),
    );
    let gather_layout = BindGroupLayoutDescriptor::new(
        "cascades_gather_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec4>(false), // rays
                storage_buffer::<Vec4>(false),           // sh_probes
                storage_buffer_read_only::<Vec4>(false), // subsamples
            ),
        ),
    );

    let pipeline = |label: String,
                    group1: &BindGroupLayoutDescriptor,
                    shader: &Handle<Shader>,
                    entry: String| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![common_layout(), group1.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let light_shader = asset_server.load(LIGHT_SHADER_PATH);
    let trace_shader = asset_server.load(TRACE_SHADER_PATH);
    let gather_shader = asset_server.load(GATHER_SHADER_PATH);
    let cascades = 0..layout::CASCADES;

    commands.insert_resource(CascadesPipeline {
        _trace_library: asset_server.load(TRACE_LIBRARY_PATH),
        light_pipeline: pipeline(
            "cascades_surface_light".into(),
            &light_layout,
            &light_shader,
            "surface_light".into(),
        ),
        trace_pipelines: cascades
            .clone()
            .map(|c| {
                pipeline(
                    format!("cascades_trace{c}"),
                    &trace_layout,
                    &trace_shader,
                    format!("cascade_{c}"),
                )
            })
            .collect(),
        gather_pipelines: cascades
            .map(|c| {
                pipeline(
                    format!("cascades_gather{c}"),
                    &gather_layout,
                    &gather_shader,
                    format!("gather_{c}"),
                )
            })
            .collect(),
        light_layout,
        trace_layout,
        gather_layout,
    });
}

fn prepare_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    layout: Res<CascadesLayout>,
    sun: Res<SunParams>,
    heightmap: Option<Res<Heightmap>>,
    existing: Option<Res<CascadesRenderData>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(heightmap) = heightmap else {
        return;
    };
    let image = |handle: &Handle<Image>| gpu_images.get(handle).map(|i| &i.texture_view);
    let (
        Some(heights),
        Some(normal_min),
        Some(normal_max),
        Some(albedo_min),
        Some(albedo_max),
        Some(radiance_min),
        Some(radiance_max),
    ) = (
        image(&heightmap.texture),
        image(&heightmap.metadata_min),
        image(&heightmap.metadata_max),
        image(&heightmap.albedo_min),
        image(&heightmap.albedo_max),
        image(&heightmap.radiance_min),
        image(&heightmap.radiance_max),
    )
    else {
        return;
    };
    let layout = &layout.layout;

    let (rays_buffer, sh_probes_buffer, subsamples_buffer) = match &existing {
        Some(e) => (
            e.rays_buffer.clone(),
            e.sh_probes_buffer.clone(),
            e.subsamples_buffer.clone(),
        ),
        None => {
            let probes: u64 = layout
                .cascades
                .iter()
                .map(|c| (c.size.x * c.size.y) as u64)
                .sum();
            let storage = |label: &'static str, vec4s: u64| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size: vec4s * 16,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let subsamples: Vec<u8> = layout
                .subsamples
                .iter()
                .flat_map(|v| v.to_array())
                .flat_map(f32::to_ne_bytes)
                .collect();
            (
                storage("cascades_rays", layout.hit_words),
                storage("cascades_sh_probes", probes * SH_STRIDE),
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("cascades_subsamples"),
                    contents: &subsamples,
                    usage: BufferUsages::STORAGE,
                }),
            )
        }
    };

    let mut cascades = [CascadeInfoUniform::default(); layout::CASCADES as usize];
    for (info, c) in cascades.iter_mut().zip(&layout.cascades) {
        *info = CascadeInfoUniform {
            size: c.size,
            interval: Vec2::from(c.interval),
            ray_base: c.hit_base,
            probe_base: c.sh_base,
            spacing: c.spacing,
            octa_side: c.octa_side,
            sub: c.sub,
            sub_base: c.sub_base,
            ..default()
        };
    }
    let uniforms = CascadesUniforms {
        hmap_min: heightmap.min_bounds,
        hmap_max: heightmap.max_bounds,
        hmap_res: heightmap.resolution,
        cascade_count: layout.cascades.len() as u32,
        hmap_top: heightmap.top,
        sun_toward: sun.toward,
        sun_irradiance: sun.irradiance,
        cascades,
    };
    let mut bytes = encase::UniformBuffer::new(Vec::new());
    bytes.write(&uniforms).unwrap();
    let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("cascades_uniforms"),
        contents: bytes.as_ref(),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group =
        |label: &'static str, layout: &BindGroupLayoutDescriptor, entries: &[BindGroupEntry]| {
            render_device.create_bind_group(
                label,
                &pipeline_cache.get_bind_group_layout(layout),
                entries,
            )
        };
    let common_bind_group = bind_group(
        "cascades_common",
        &common_layout(),
        &BindGroupEntries::sequential((
            heights,
            uniform_buffer.as_entire_binding(),
            normal_min,
            normal_max,
        )),
    );
    let light_bind_group = bind_group(
        "cascades_light",
        &pipeline.light_layout,
        &BindGroupEntries::sequential((
            normal_min,
            normal_max,
            albedo_min,
            albedo_max,
            radiance_min,
            radiance_max,
        )),
    );
    let trace_bind_group = bind_group(
        "cascades_trace",
        &pipeline.trace_layout,
        &BindGroupEntries::sequential((
            radiance_min,
            radiance_max,
            rays_buffer.as_entire_binding(),
        )),
    );
    let gather_bind_group = bind_group(
        "cascades_gather",
        &pipeline.gather_layout,
        &BindGroupEntries::sequential((
            rays_buffer.as_entire_binding(),
            sh_probes_buffer.as_entire_binding(),
            subsamples_buffer.as_entire_binding(),
        )),
    );

    commands.insert_resource(CascadesRenderData {
        rays_buffer,
        sh_probes_buffer,
        subsamples_buffer,
        common_bind_group,
        light_bind_group,
        trace_bind_group,
        gather_bind_group,
    });
}

fn dispatch_cascades(
    layout: Res<CascadesLayout>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CascadesPipeline>,
    heightmap: Option<Res<Heightmap>>,
    render_data: Option<Res<CascadesRenderData>>,
    mut render_context: RenderContext,
) {
    let (Some(render_data), Some(heightmap)) = (render_data, heightmap) else {
        return;
    };
    // Every pass reads the previous one's output; skip the frame until all
    // have compiled.
    let get = |id: &CachedComputePipelineId| pipeline_cache.get_compute_pipeline(*id);
    let (Some(light), Some(traces), Some(gathers)) = (
        get(&pipeline.light_pipeline),
        pipeline
            .trace_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
        pipeline
            .gather_pipelines
            .iter()
            .map(get)
            .collect::<Option<Vec<_>>>(),
    ) else {
        return;
    };

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("cascades2_5d"),
            ..default()
        });
    pass.set_bind_group(0, &render_data.common_bind_group, &[]);

    pass.set_pipeline(light);
    pass.set_bind_group(1, &render_data.light_bind_group, &[]);
    pass.dispatch_workgroups(
        heightmap.resolution.x.div_ceil(LIGHT_WORKGROUP),
        heightmap.resolution.y.div_ceil(LIGHT_WORKGROUP),
        1,
    );

    // Top-down: each cascade merges the one above it.
    pass.set_bind_group(1, &render_data.trace_bind_group, &[]);
    for (c, trace) in layout.layout.cascades.iter().zip(&traces).rev() {
        pass.set_pipeline(trace);
        pass.dispatch_workgroups((c.rays() as u32).div_ceil(WORKGROUP), 1, 1);
    }

    pass.set_bind_group(1, &render_data.gather_bind_group, &[]);
    for (c, gather) in layout.layout.cascades.iter().zip(&gathers) {
        pass.set_pipeline(gather);
        pass.dispatch_workgroups((c.size.x * c.size.y).div_ceil(WORKGROUP), 1, 1);
    }
}
