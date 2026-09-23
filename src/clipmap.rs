//! SDF geometry grid: one camera-centered window of RESOLUTION^3 voxels at
//! VOXEL (no tiers, no LOD), a 3D texture mapped toroidally: world voxel v
//! lives in texel v mod RESOLUTION. A sampler with repeat addressing then
//! wraps trilinear filtering across the seam for free. Beyond the window is
//! empty space.
//!
//! Slice-triggered updates (ARCHITECTURE.md): when the camera crosses a
//! voxel boundary, only the newly revealed slabs are recomposed on the GPU,
//! from the per-mesh SDF grids baked by `sdf_bake` (Lumen-style mesh
//! distance fields). Everything else stays as it was.
//!
//! Besides distance, the grid holds, for voxels near surfaces:
//! - albedo, from the nearest mesh's material;
//! - unshadowed surface color: albedo / pi * sun irradiance * cos, with the
//!   normal from the nearest mesh's own baked grid;
//! - sun visibility: the shadow clipmap (Aaltonen's estimator, marched once
//!   per voxel when its region is composed).
//! Radiance leaving a surface is color * visibility. Keeping visibility
//! separate is what lets it be re-marched on its own (time of day,
//! double-buffered) without touching the rest.
//!
//! `K` throws the grid away and recomposes it, for checking that the
//! scrolled result matches a fresh one.

use std::path::Path;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{sampler, storage_buffer_read_only, texture_3d, texture_storage_3d, uniform_buffer},
            *,
        },
        diagnostic::RecordDiagnostics,
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use crate::scene3d::SceneBoxMesh;
use crate::sdf_bake::{self, SdfGrid};

/// Voxels per axis: 256 at 12.5 cm spans 32 m, +-16 m around the camera.
pub const RESOLUTION: i32 = 256;
pub const VOXEL: f32 = 0.125;
pub const FORMAT: TextureFormat = TextureFormat::R16Float;
pub const ALBEDO_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
pub const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const SHADOW_FORMAT: TextureFormat = TextureFormat::R16Float;
/// Per-mesh bake resolution: half the grid's voxel.
const BAKE_VOXEL: f32 = VOXEL / 2.0;
const BAKE_PADDING: u32 = 3;
const COMPOSE_SHADER: &str = "shaders/clipmap_compose.wgsl";
const SHADOW_SHADER: &str = "shaders/clipmap_shadow.wgsl";
const COMPOSE_SPAN: &str = "clipmap_compose";
const SHADOW_SPAN: &str = "clipmap_shadow";
/// Shared tracing code (`s_rc::clipmap_sample`), imported by every shader
/// that traces the clipmap; must stay loaded for those imports to resolve.
const SAMPLE_SHADER: &str = "shaders/clipmap_sample.wgsl";

#[derive(Resource)]
struct ClipmapShaderLibrary(#[allow(dead_code)] Handle<Shader>);

pub struct ClipmapPlugin;

/// Main-world clipmap bookkeeping; run things that read the window origins
/// after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClipmapSystems;

/// Render-world clipmap composition; tracing the clipmap goes after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClipmapRenderSystems;

/// The sun, as the clipmap and everything shading from it need it.
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct SunLight {
    /// Unit vector toward the sun.
    pub toward: Vec3,
    pub irradiance: Vec3,
    /// Cone factor for soft shadows: 1 / tan(angular radius).
    pub k: f32,
}

/// The grid's window at this frame, as the shaders' `Clipmap`.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct ClipmapUniform {
    pub origin: IVec4,
    pub voxel: f32,
    pub resolution: u32,
}

impl ClipmapUniform {
    pub fn from_state(state: &ClipmapState) -> Self {
        Self { origin: state.origin.unwrap_or_default().extend(0), voxel: VOXEL, resolution: RESOLUTION as u32 }
    }
}

/// The camera the clipmap follows.
#[derive(Component)]
pub struct ClipmapCamera;

/// Baked per-mesh grids and their instances, uploaded once.
#[derive(Resource, Clone, ExtractResource)]
pub struct SdfScene {
    pub instances: Vec<GpuInstance>,
    pub values: Vec<f32>,
}

#[derive(ShaderType, Clone, Copy, Debug)]
pub struct GpuInstance {
    pub world_to_local: Mat4,
    pub grid_min: Vec3,
    pub voxel: f32,
    pub dims: UVec3,
    pub offset: u32,
    /// Linear albedo in rgb.
    pub albedo: Vec4,
}

/// An axis-aligned block of world voxels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub min: IVec3,
    pub size: UVec3,
}

impl Region {
    pub fn voxels(&self) -> u64 {
        self.size.x as u64 * self.size.y as u64 * self.size.z as u64
    }
}

/// The window origin (world voxel coordinates of its min corner) and the
/// regions to recompose this frame.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct ClipmapState {
    pub origin: Option<IVec3>,
    pub pending: Vec<Region>,
    /// Voxels recomposed since the last log line.
    pub updated_since_log: u64,
    /// Bumping this recomposes the whole grid from scratch.
    pub generation: u32,
    last_generation: Option<u32>,
}

impl ClipmapState {
    pub fn window_origin(camera: Vec3) -> IVec3 {
        (camera / VOXEL).floor().as_ivec3() - IVec3::splat(RESOLUTION / 2)
    }
}

/// Regions of the window at `new` that weren't in the window at `old`
/// (both `res` voxels per axis): the full window if there is no old one or
/// they don't overlap, else up to one slab per axis that moved. Slabs may
/// overlap at edges and corners, which only means a few voxels get
/// recomposed twice.
pub fn revealed_regions(old: Option<IVec3>, new: IVec3, res: i32) -> Vec<Region> {
    let full = Region { min: new, size: UVec3::splat(res as u32) };
    let Some(old) = old else { return vec![full] };
    let delta = new - old;
    if delta == IVec3::ZERO {
        return Vec::new();
    }
    if delta.abs().max_element() >= res {
        return vec![full];
    }
    let mut regions = Vec::new();
    for axis in 0..3 {
        let d = delta[axis];
        if d == 0 {
            continue;
        }
        let mut min = new;
        let mut size = UVec3::splat(res as u32);
        size[axis] = d.unsigned_abs();
        // Moving +: the new layers are at the far end of the new window.
        min[axis] = if d > 0 { old[axis] + res } else { new[axis] };
        regions.push(Region { min, size });
    }
    regions
}

impl Plugin for ClipmapPlugin {
    fn build(&self, app: &mut App) {
        let library = app.world().resource::<AssetServer>().load(SAMPLE_SHADER);
        app.insert_resource(ClipmapShaderLibrary(library))
            .init_resource::<ClipmapState>()
            .add_plugins((
                ExtractResourcePlugin::<SdfScene>::default(),
                ExtractResourcePlugin::<ClipmapState>::default(),
                ExtractResourcePlugin::<SunLight>::default(),
            ))
            .add_systems(
                PostUpdate,
                (bake_scene, rebuild_on_key, update_clipmap_state, log_clipmap_updates)
                    .chain()
                    .in_set(ClipmapSystems)
                    .after(TransformSystems::Propagate),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_clipmap_gpu)
            .add_systems(Render, upload_sdf_scene.in_set(RenderSystems::PrepareResources))
            .add_systems(
                Core3d,
                compose_clipmap.in_set(ClipmapRenderSystems).in_set(Core3dSystems::Prepass),
            );
    }
}

/// Once the scene's box meshes exist: bake (or load from cache) each mesh's
/// SDF grid in mesh-local space, and instance it with its transform.
fn bake_scene(
    mut commands: Commands,
    existing: Option<Res<SdfScene>>,
    boxes: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>, &GlobalTransform), With<SceneBoxMesh>>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
) {
    if existing.is_some() || boxes.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let cache = Path::new("target/sdf_cache");
    let mut scene = SdfScene { instances: Vec::new(), values: Vec::new() };
    for (mesh, material, transform) in &boxes {
        let (Some(mesh), Some(material)) = (meshes.get(&mesh.0), materials.get(&material.0)) else { return };
        let albedo = material.base_color.to_linear();
        let grid: SdfGrid = match sdf_bake::mesh_triangles(mesh)
            .and_then(|(p, i)| sdf_bake::bake_cached(&p, &i, BAKE_VOXEL, BAKE_PADDING, cache))
        {
            Ok(grid) => grid,
            Err(e) => {
                error!("SDF bake failed: {e}");
                return;
            }
        };
        scene.instances.push(GpuInstance {
            world_to_local: transform.to_matrix().inverse(),
            grid_min: grid.min,
            voxel: grid.voxel,
            dims: grid.dims,
            offset: scene.values.len() as u32,
            albedo: Vec4::new(albedo.red, albedo.green, albedo.blue, 0.0),
        });
        scene.values.extend_from_slice(&grid.values);
    }
    info!(
        "SDF scene: {} instances, {} voxels ({:.1} MB), baked/loaded in {:.2}s",
        scene.instances.len(),
        scene.values.len(),
        scene.values.len() as f64 * 4.0 / 1e6,
        started.elapsed().as_secs_f64()
    );
    commands.insert_resource(scene);
}

fn rebuild_on_key(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<ClipmapState>) {
    if keys.just_pressed(KeyCode::KeyK) {
        state.generation += 1;
        info!("clipmap: full rebuild");
    }
}

fn update_clipmap_state(
    mut state: ResMut<ClipmapState>,
    scene: Option<Res<SdfScene>>,
    camera: Query<&GlobalTransform, With<ClipmapCamera>>,
) {
    let (Some(_), Ok(camera)) = (scene, camera.single()) else {
        return;
    };
    let state = &mut *state;
    if state.last_generation != Some(state.generation) {
        state.origin = None;
        state.last_generation = Some(state.generation);
    }
    let origin = ClipmapState::window_origin(camera.translation());
    let regions = revealed_regions(state.origin, origin, RESOLUTION);
    state.updated_since_log += regions.iter().map(Region::voxels).sum::<u64>();
    state.pending = regions;
    state.origin = Some(origin);
}

fn log_clipmap_updates(time: Res<Time>, mut timer: Local<Option<Timer>>, mut state: ResMut<ClipmapState>) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }
    let voxels = state.updated_since_log;
    info!("grid voxels recomposed in the last 2s: {voxels} ({:.2} windows)", voxels as f64 / RESOLUTION.pow(3) as f64);
    state.updated_since_log = 0;
}

// ---------------------------------------------------------------- render

/// The grid's textures, shared with everything that traces or shades from
/// it.
#[derive(Resource)]
pub struct ClipmapTextures {
    pub sdf: TextureView,
    pub albedo: TextureView,
    pub color: TextureView,
    pub shadow: TextureView,
    pub sampler: Sampler,
}

#[derive(ShaderType, Clone, Copy)]
struct ComposeParams {
    region_min: IVec3,
    instance_count: u32,
    region_size: UVec3,
    resolution: u32,
    sun_toward: Vec3,
    voxel: f32,
    sun_irradiance: Vec3,
}

#[derive(ShaderType, Clone, Copy)]
struct ShadowParams {
    clipmap: ClipmapUniform,
    region_min: IVec3,
    sun_k: f32,
    region_size: UVec3,
    sun_toward: Vec3,
}

#[derive(Resource)]
struct ComposeGpu {
    compose_layout: BindGroupLayoutDescriptor,
    compose: CachedComputePipelineId,
    shadow_layout: BindGroupLayoutDescriptor,
    shadow: CachedComputePipelineId,
    storage: TextureView,
    albedo_storage: TextureView,
    color_storage: TextureView,
    shadow_storage: TextureView,
}

#[derive(Resource)]
struct SdfSceneBuffers {
    instances: Buffer,
    values: Buffer,
    instance_count: u32,
}

fn grid_texture(render_device: &RenderDevice, name: &str, format: TextureFormat) -> TextureView {
    render_device
        .create_texture(&TextureDescriptor {
            label: Some(&format!("grid_{name}")),
            size: Extent3d {
                width: RESOLUTION as u32,
                height: RESOLUTION as u32,
                depth_or_array_layers: RESOLUTION as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

fn init_clipmap_gpu(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let sdf = grid_texture(&render_device, "sdf", FORMAT);
    let albedo = grid_texture(&render_device, "albedo", ALBEDO_FORMAT);
    let color = grid_texture(&render_device, "color", COLOR_FORMAT);
    let shadow = grid_texture(&render_device, "shadow", SHADOW_FORMAT);
    let linear_repeat = render_device.create_sampler(&SamplerDescriptor {
        label: Some("clipmap_sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    let compose_layout = BindGroupLayoutDescriptor::new(
        "clipmap_compose_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ComposeParams>(true),
                storage_buffer_read_only::<GpuInstance>(false),
                storage_buffer_read_only::<f32>(false),
                texture_storage_3d(FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_3d(ALBEDO_FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_3d(COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shadow_layout = BindGroupLayoutDescriptor::new(
        "clipmap_shadow_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ShadowParams>(true),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_3d(SHADOW_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = |label: &'static str, layout: &BindGroupLayoutDescriptor, shader: &'static str, entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader: asset_server.load(shader),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let compose = pipeline("clipmap_compose", &compose_layout, COMPOSE_SHADER, "compose");
    let shadow_pipeline = pipeline("clipmap_shadow", &shadow_layout, SHADOW_SHADER, "shadow");

    commands.insert_resource(ComposeGpu {
        compose_layout,
        compose,
        shadow_layout,
        shadow: shadow_pipeline,
        storage: sdf.clone(),
        albedo_storage: albedo.clone(),
        color_storage: color.clone(),
        shadow_storage: shadow.clone(),
    });
    commands.insert_resource(ClipmapTextures { sdf, albedo, color, shadow, sampler: linear_repeat });
}

fn upload_sdf_scene(
    mut commands: Commands,
    scene: Option<Res<SdfScene>>,
    existing: Option<Res<SdfSceneBuffers>>,
    render_device: Res<RenderDevice>,
) {
    let Some(scene) = scene else { return };
    if existing.is_some() && !scene.is_changed() {
        return;
    }
    let mut instance_bytes = encase::StorageBuffer::new(Vec::new());
    instance_bytes.write(&scene.instances).unwrap();
    commands.insert_resource(SdfSceneBuffers {
        instances: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sdf_instances"),
            contents: instance_bytes.as_ref(),
            usage: BufferUsages::STORAGE,
        }),
        values: render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sdf_values"),
            contents: bytemuck_f32(&scene.values),
            usage: BufferUsages::STORAGE,
        }),
        instance_count: scene.instances.len() as u32,
    });
}

fn bytemuck_f32(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

#[allow(clippy::too_many_arguments)]
fn compose_clipmap(
    state: Res<ClipmapState>,
    sun: Option<Res<SunLight>>,
    compose: Option<Res<ComposeGpu>>,
    buffers: Option<Res<SdfSceneBuffers>>,
    textures: Option<Res<ClipmapTextures>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut compose_params: Local<DynamicUniformBuffer<ComposeParams>>,
    mut shadow_params: Local<DynamicUniformBuffer<ShadowParams>>,
    mut composed_generation: Local<Option<u32>>,
    mut ctx: RenderContext,
) {
    let (Some(sun), Some(compose), Some(buffers), Some(textures)) = (sun, compose, buffers, textures) else {
        return;
    };
    for id in [compose.compose, compose.shadow] {
        if let CachedPipelineState::Err(e) = pipeline_cache.get_compute_pipeline_state(id) {
            error_once!("clipmap pipeline failed: {e}");
        }
    }
    let (Some(compose_pipeline), Some(shadow_pipeline)) = (
        pipeline_cache.get_compute_pipeline(compose.compose),
        pipeline_cache.get_compute_pipeline(compose.shadow),
    ) else {
        return;
    };
    // Until a full window has been composed for this generation (e.g. the
    // pipelines were still compiling when the main world asked for it), the
    // incremental regions aren't enough: compose the whole window instead.
    let full = *composed_generation != Some(state.generation);
    let regions = match (full, state.origin) {
        (true, Some(origin)) => vec![Region { min: origin, size: UVec3::splat(RESOLUTION as u32) }],
        _ => state.pending.clone(),
    };
    if regions.is_empty() {
        return;
    }
    if full {
        *composed_generation = Some(state.generation);
    }

    compose_params.clear();
    shadow_params.clear();
    let clipmap = ClipmapUniform::from_state(&state);
    let mut jobs = Vec::new();
    for region in &regions {
        let compose_offset = compose_params.push(&ComposeParams {
            region_min: region.min,
            instance_count: buffers.instance_count,
            region_size: region.size,
            resolution: RESOLUTION as u32,
            sun_toward: sun.toward,
            voxel: VOXEL,
            sun_irradiance: sun.irradiance,
        });
        let shadow_offset = shadow_params.push(&ShadowParams {
            clipmap,
            region_min: region.min,
            sun_k: sun.k,
            region_size: region.size,
            sun_toward: sun.toward,
        });
        jobs.push((*region, compose_offset, shadow_offset));
    }
    compose_params.write_buffer(&render_device, &render_queue);
    shadow_params.write_buffer(&render_device, &render_queue);
    let (Some(compose_binding), Some(shadow_binding)) = (compose_params.binding(), shadow_params.binding()) else {
        return;
    };

    let compose_layout = pipeline_cache.get_bind_group_layout(&compose.compose_layout);
    let compose_group = render_device.create_bind_group(
        "clipmap_compose",
        &compose_layout,
        &BindGroupEntries::sequential((
            compose_binding,
            buffers.instances.as_entire_buffer_binding(),
            buffers.values.as_entire_buffer_binding(),
            &compose.storage,
            &compose.albedo_storage,
            &compose.color_storage,
        )),
    );
    let shadow_layout = pipeline_cache.get_bind_group_layout(&compose.shadow_layout);
    let shadow_group = render_device.create_bind_group(
        "clipmap_shadow",
        &shadow_layout,
        &BindGroupEntries::sequential((shadow_binding, &textures.sdf, &textures.sampler, &compose.shadow_storage)),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let descriptor = ComputePassDescriptor { label: Some("clipmap"), timestamp_writes: None };
    // Geometry (and color) for every region first, then shadows: a shadow
    // march reads the geometry around it, which may be in another region.
    {
        let mut pass = ctx.command_encoder().begin_compute_pass(&descriptor);
        let span = diagnostics.pass_span(&mut pass, COMPOSE_SPAN);
        pass.set_pipeline(compose_pipeline);
        for (region, offset, _) in &jobs {
            pass.set_bind_group(0, &compose_group, &[*offset]);
            let groups = region.size.map(|s| s.div_ceil(4));
            pass.dispatch_workgroups(groups.x, groups.y, groups.z);
        }
        span.end(&mut pass);
    }
    {
        let mut pass = ctx.command_encoder().begin_compute_pass(&descriptor);
        let span = diagnostics.pass_span(&mut pass, SHADOW_SPAN);
        pass.set_pipeline(shadow_pipeline);
        for (region, _, offset) in &jobs {
            pass.set_bind_group(0, &shadow_group, &[*offset]);
            let groups = region.size.map(|s| s.div_ceil(4));
            pass.dispatch_workgroups(groups.x, groups.y, groups.z);
        }
        span.end(&mut pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn voxels(regions: &[Region]) -> HashSet<IVec3> {
        let mut set = HashSet::new();
        for r in regions {
            for z in 0..r.size.z as i32 {
                for y in 0..r.size.y as i32 {
                    for x in 0..r.size.x as i32 {
                        set.insert(r.min + IVec3::new(x, y, z));
                    }
                }
            }
        }
        set
    }

    fn window(origin: IVec3, res: i32) -> HashSet<IVec3> {
        voxels(&[Region { min: origin, size: UVec3::splat(res as u32) }])
    }

    /// The updated regions must be exactly the voxels in the new window
    /// that weren't in the old one, for every kind of move.
    #[test]
    fn revealed_regions_are_exactly_the_new_voxels() {
        let res = 6;
        let old = IVec3::new(2, -3, 5);
        for dx in -7..=7 {
            for dy in [-7, -2, 0, 1, 6] {
                for dz in [-1, 0, 3] {
                    let new = old + IVec3::new(dx, dy, dz);
                    let got = voxels(&revealed_regions(Some(old), new, res));
                    let expected: HashSet<IVec3> =
                        window(new, res).difference(&window(old, res)).copied().collect();
                    assert_eq!(got, expected, "move {:?}", new - old);
                }
            }
        }
        assert_eq!(voxels(&revealed_regions(None, old, res)), window(old, res));
    }
}
