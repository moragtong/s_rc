//! Debug view of the SDF clipmap: sphere-traces it from the camera at a
//! fixed 800x450 and shows the result full-window.
//!
//! Controls: `V` toggles the view, `N` cycles normals / tier / step count,
//! `X` exports the scene, camera and traced depth to `target/rc3d/` for
//! `tools/mitsuba_depth.py` to compare against Mitsuba.

use std::path::Path;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{sampler, storage_buffer_sized, texture_3d, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use crate::clipmap::{ClipmapCamera, ClipmapRenderSystems, ClipmapState, ClipmapTextures, RESOLUTION, TIER_VOXEL, TIERS};
use crate::scene3d::Scene3d;

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 450;
const SHADER: &str = "shaders/sdf_view.wgsl";
const EXPORT_DIR: &str = "target/rc3d";

pub struct SdfViewPlugin;

#[derive(Resource, Clone, ExtractResource)]
pub struct SdfView {
    pub enabled: bool,
    pub mode: u32,
    pub image: Handle<Image>,
    pub depth: Handle<ShaderBuffer>,
    pub uniform: SdfViewUniform,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub struct SdfViewUniform {
    camera: Vec3,
    width: u32,
    right: Vec3,
    height: u32,
    up: Vec3,
    mode: u32,
    forward: Vec3,
    resolution: u32,
    origins: [IVec4; TIERS],
    voxels: Vec4,
}

/// Marks the full-window UI node showing the traced image.
#[derive(Component)]
struct SdfViewNode;

impl Plugin for SdfViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SdfView>::default())
            .add_systems(Startup, setup_sdf_view)
            .add_systems(Update, sdf_view_input)
            .add_systems(PostUpdate, update_sdf_view_uniform.after(crate::clipmap::ClipmapSystems));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_sdf_view_pipeline)
            .add_systems(Core3d, trace_sdf_view.after(ClipmapRenderSystems).in_set(Core3dSystems::Prepass));
    }
}

fn setup_sdf_view(mut commands: Commands, mut images: ResMut<Assets<Image>>, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let mut image = Image::new_fill(
        Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    let image = images.add(image);
    let depth = buffers.add(ShaderBuffer::with_size((WIDTH * HEIGHT * 4) as usize, RenderAssetUsages::RENDER_WORLD));

    commands.spawn((
        Node { width: percent(100), height: percent(100), ..default() },
        ImageNode::new(image.clone()),
        Visibility::Hidden,
        SdfViewNode,
    ));
    commands.insert_resource(SdfView { enabled: false, mode: 0, image, depth, uniform: default() });
}

fn sdf_view_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut view: ResMut<SdfView>,
    mut node: Query<&mut Visibility, With<SdfViewNode>>,
    mut commands: Commands,
    scene: Option<Res<SceneResource>>,
    state: Res<ClipmapState>,
    camera: Query<(&GlobalTransform, &Projection), With<ClipmapCamera>>,
) {
    if keys.just_pressed(KeyCode::KeyV) {
        view.enabled = !view.enabled;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        view.mode = (view.mode + 1) % 3;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        // The trace only runs while the view is on.
        view.enabled = true;
        if let (Some(scene), Ok((transform, projection))) = (scene, camera.single()) {
            export(&scene.0, transform, projection, &state, &view, &mut commands);
        }
    }
    for mut visibility in &mut node {
        *visibility = if view.enabled { Visibility::Visible } else { Visibility::Hidden };
    }
}

/// The scene definition, kept for exports.
#[derive(Resource)]
pub struct SceneResource(pub Scene3d);

fn fov_y(projection: &Projection) -> f32 {
    match projection {
        Projection::Perspective(p) => p.fov,
        _ => std::f32::consts::FRAC_PI_4,
    }
}

fn update_sdf_view_uniform(
    mut view: ResMut<SdfView>,
    state: Res<ClipmapState>,
    camera: Query<(&GlobalTransform, &Projection), With<ClipmapCamera>>,
) {
    let Ok((transform, projection)) = camera.single() else { return };
    let tan = (fov_y(projection) / 2.0).tan();
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let origins = std::array::from_fn(|t| state.origins[t].unwrap_or_default().extend(0));
    let mode = view.mode;
    view.uniform = SdfViewUniform {
        camera: transform.translation(),
        width: WIDTH,
        right: transform.right().as_vec3() * tan * aspect,
        height: HEIGHT,
        up: transform.up().as_vec3() * tan,
        mode,
        forward: transform.forward().as_vec3(),
        resolution: RESOLUTION as u32,
        origins,
        voxels: Vec4::new(TIER_VOXEL[0], TIER_VOXEL[1], TIER_VOXEL[2], 0.0),
    };
}

/// Writes the scene and camera as JSON, and the traced depth (f32 per
/// pixel, row-major, top row first; -1 = miss) once it's read back.
fn export(
    scene: &Scene3d,
    transform: &GlobalTransform,
    projection: &Projection,
    state: &ClipmapState,
    view: &SdfView,
    commands: &mut Commands,
) {
    let dir = Path::new(EXPORT_DIR);
    if let Err(e) = std::fs::create_dir_all(dir) {
        error!("export failed: {e}");
        return;
    }
    let camera = crate::scene3d::ExportCamera {
        position: transform.translation(),
        forward: transform.forward().as_vec3(),
        up: transform.up().as_vec3(),
        fov_y_degrees: fov_y(projection).to_degrees(),
        width: WIDTH,
        height: HEIGHT,
        tiers: (0..TIERS)
            .map(|t| (state.origins[t].unwrap_or_default(), TIER_VOXEL[t], RESOLUTION))
            .collect(),
    };
    if let Err(e) = std::fs::write(dir.join("scene.json"), scene.export_json(&camera)) {
        error!("export failed: {e}");
        return;
    }
    commands.spawn(Readback::buffer(view.depth.clone())).observe(
        |event: On<ReadbackComplete>, mut commands: Commands, mut handled: Local<bool>| {
            if std::mem::replace(&mut *handled, true) {
                return;
            }
            commands.entity(event.entity).try_despawn();
            let path = Path::new(EXPORT_DIR).join("sdf_depth.f32");
            match std::fs::write(&path, &event.data) {
                Ok(()) => {
                    let hits = event
                        .data
                        .chunks_exact(4)
                        // Misses are -1 and hits lie beyond the camera; an unwritten
                        // (all-zero) buffer must not count as hits.
                        .filter(|b| f32::from_le_bytes((*b).try_into().unwrap()) > 0.0)
                        .count();
                    info!(
                        "exported {EXPORT_DIR}/scene.json and {} ({hits} of {} pixels hit)",
                        path.display(),
                        WIDTH * HEIGHT
                    );
                }
                Err(e) => error!("export failed: {e}"),
            }
        },
    );
}

// ---------------------------------------------------------------- render

#[derive(Resource)]
struct SdfViewPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_sdf_view_pipeline(mut commands: Commands, asset_server: Res<AssetServer>, pipeline_cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "sdf_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<SdfViewUniform>(false),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                storage_buffer_sized(false, std::num::NonZero::new((WIDTH * HEIGHT * 4) as u64)),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sdf_view".into()),
        layout: vec![layout.clone()],
        shader: asset_server.load(SHADER),
        entry_point: Some("trace".into()),
        ..default()
    });
    commands.insert_resource(SdfViewPipeline { layout, pipeline });
}

#[allow(clippy::too_many_arguments)]
fn trace_sdf_view(
    view: Res<SdfView>,
    pipeline: Res<SdfViewPipeline>,
    textures: Option<Res<ClipmapTextures>>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    if !view.enabled {
        return;
    }
    let (Some(textures), Some(compute), Some(image), Some(depth)) = (
        textures,
        pipeline_cache.get_compute_pipeline(pipeline.pipeline),
        gpu_images.get(&view.image),
        gpu_buffers.get(&view.depth),
    ) else {
        return;
    };
    let mut uniform = UniformBuffer::from(view.uniform);
    uniform.write_buffer(&render_device, &render_queue);
    let Some(uniform_binding) = uniform.binding() else { return };
    let bind_group = render_device.create_bind_group(
        "sdf_view",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            uniform_binding,
            &textures.views[0],
            &textures.views[1],
            &textures.views[2],
            &textures.sampler,
            &image.texture_view,
            depth.buffer.as_entire_buffer_binding(),
        )),
    );
    let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
        label: Some("sdf_view"),
        timestamp_writes: None,
    });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);
}
