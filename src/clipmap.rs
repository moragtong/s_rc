//! SDF geometry clipmap: three camera-centered tiers of 128^3 voxels at
//! 12.5 cm / 50 cm / 2 m (spanning 16 / 64 / 256 m), each a 3D texture
//! mapped toroidally: world voxel v lives in texel v mod 128. A sampler with
//! repeat addressing then wraps trilinear filtering across the seam for free.
//!
//! Slice-triggered updates (ARCHITECTURE.md): when the camera crosses a
//! tier's voxel boundary, only the newly revealed slabs are recomposed on
//! the GPU, from the per-mesh SDF grids baked by `sdf_bake` (Lumen-style
//! mesh distance fields). Everything else stays as it was.
//!
//! `K` throws the clipmap away and recomposes every tier, for checking
//! that the scrolled result matches a fresh one.

use std::path::Path;

use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{storage_buffer_read_only, texture_storage_3d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use crate::scene3d::SceneBoxMesh;
use crate::sdf_bake::{self, SdfGrid};

pub const TIERS: usize = 3;
pub const RESOLUTION: i32 = 128;
pub const TIER_VOXEL: [f32; TIERS] = [0.125, 0.5, 2.0];
pub const FORMAT: TextureFormat = TextureFormat::R16Float;
/// Per-mesh bake resolution: half the finest tier's voxel.
const BAKE_VOXEL: f32 = 0.0625;
const BAKE_PADDING: u32 = 3;
const COMPOSE_SHADER: &str = "shaders/clipmap_compose.wgsl";
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
}

/// An axis-aligned block of world voxels (in one tier's voxel units).
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

/// Per-tier window origins (world voxel coordinates of each window's min
/// corner) and the regions to recompose this frame.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct ClipmapState {
    pub origins: [Option<IVec3>; TIERS],
    pub pending: [Vec<Region>; TIERS],
    /// Voxels recomposed per tier since the last log line.
    pub updated_since_log: [u64; TIERS],
    /// Bumping this recomposes every tier from scratch.
    pub generation: u32,
    last_generation: Option<u32>,
}

impl ClipmapState {
    pub fn tier_origin(tier: usize, camera: Vec3) -> IVec3 {
        (camera / TIER_VOXEL[tier]).floor().as_ivec3() - IVec3::splat(RESOLUTION / 2)
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
    boxes: Query<(&Mesh3d, &GlobalTransform), With<SceneBoxMesh>>,
    meshes: Res<Assets<Mesh>>,
) {
    if existing.is_some() || boxes.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let cache = Path::new("target/sdf_cache");
    let mut scene = SdfScene { instances: Vec::new(), values: Vec::new() };
    for (mesh, transform) in &boxes {
        let Some(mesh) = meshes.get(&mesh.0) else { return };
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
        state.origins = [None; TIERS];
        state.last_generation = Some(state.generation);
    }
    for tier in 0..TIERS {
        let origin = ClipmapState::tier_origin(tier, camera.translation());
        let regions = revealed_regions(state.origins[tier], origin, RESOLUTION);
        state.updated_since_log[tier] += regions.iter().map(Region::voxels).sum::<u64>();
        state.pending[tier] = regions;
        state.origins[tier] = Some(origin);
    }
}

fn log_clipmap_updates(time: Res<Time>, mut timer: Local<Option<Timer>>, mut state: ResMut<ClipmapState>) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(2.0, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }
    let per_tier = RESOLUTION.pow(3) as f64;
    let parts: Vec<String> = (0..TIERS)
        .map(|t| format!("tier {t} {} ({:.1} windows)", state.updated_since_log[t], state.updated_since_log[t] as f64 / per_tier))
        .collect();
    info!("clipmap voxels recomposed in the last 2s: {}", parts.join(", "));
    state.updated_since_log = [0; TIERS];
}

// ---------------------------------------------------------------- render

/// The tier textures, shared with the SDF view.
#[derive(Resource)]
pub struct ClipmapTextures {
    pub views: [TextureView; TIERS],
    pub sampler: Sampler,
}

#[derive(ShaderType, Clone, Copy)]
struct ComposeParams {
    region_min: IVec3,
    instance_count: u32,
    region_size: UVec3,
    resolution: u32,
    voxel: f32,
}

#[derive(Resource)]
struct ComposeGpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    storage_views: [TextureView; TIERS],
}

#[derive(Resource)]
struct SdfSceneBuffers {
    instances: Buffer,
    values: Buffer,
    instance_count: u32,
}

fn init_clipmap_gpu(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let textures: [Texture; TIERS] = std::array::from_fn(|tier| {
        render_device.create_texture(&TextureDescriptor {
            label: Some(&format!("clipmap_tier_{tier}")),
            size: Extent3d {
                width: RESOLUTION as u32,
                height: RESOLUTION as u32,
                depth_or_array_layers: RESOLUTION as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: FORMAT,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    });
    let views: [TextureView; TIERS] = std::array::from_fn(|t| textures[t].create_view(&TextureViewDescriptor::default()));
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("clipmap_sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    let layout = BindGroupLayoutDescriptor::new(
        "clipmap_compose_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ComposeParams>(true),
                storage_buffer_read_only::<GpuInstance>(false),
                storage_buffer_read_only::<f32>(false),
                texture_storage_3d(FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("clipmap_compose".into()),
        layout: vec![layout.clone()],
        shader: asset_server.load(COMPOSE_SHADER),
        entry_point: Some("compose".into()),
        ..default()
    });

    commands.insert_resource(ComposeGpu { layout, pipeline, storage_views: views.clone() });
    commands.insert_resource(ClipmapTextures { views, sampler });
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
    compose: Option<Res<ComposeGpu>>,
    buffers: Option<Res<SdfSceneBuffers>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: Local<DynamicUniformBuffer<ComposeParams>>,
    mut composed_generation: Local<Option<u32>>,
    mut ctx: RenderContext,
) {
    let (Some(compose), Some(buffers)) = (compose, buffers) else { return };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(compose.pipeline) else { return };
    // Until a full window has been composed for this generation (e.g. the
    // pipeline was still compiling when the main world asked for it), the
    // incremental regions aren't enough: compose whole windows instead.
    let full = *composed_generation != Some(state.generation);
    let regions: [Vec<Region>; TIERS] = std::array::from_fn(|tier| match (full, state.origins[tier]) {
        (true, Some(origin)) => vec![Region { min: origin, size: UVec3::splat(RESOLUTION as u32) }],
        _ => state.pending[tier].clone(),
    });
    if regions.iter().all(Vec::is_empty) {
        return;
    }
    if full {
        *composed_generation = Some(state.generation);
    }

    params.clear();
    let mut jobs = Vec::new();
    for tier in 0..TIERS {
        for region in &regions[tier] {
            let offset = params.push(&ComposeParams {
                region_min: region.min,
                instance_count: buffers.instance_count,
                region_size: region.size,
                resolution: RESOLUTION as u32,
                voxel: TIER_VOXEL[tier],
            });
            jobs.push((tier, *region, offset));
        }
    }
    params.write_buffer(&render_device, &render_queue);
    let Some(params_binding) = params.binding() else { return };

    let layout = pipeline_cache.get_bind_group_layout(&compose.layout);
    let bind_groups: [BindGroup; TIERS] = std::array::from_fn(|tier| {
        render_device.create_bind_group(
            "clipmap_compose",
            &layout,
            &BindGroupEntries::sequential((
                params_binding.clone(),
                buffers.instances.as_entire_buffer_binding(),
                buffers.values.as_entire_buffer_binding(),
                &compose.storage_views[tier],
            )),
        )
    });

    let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
        label: Some("clipmap_compose"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    for (tier, region, offset) in jobs {
        pass.set_bind_group(0, &bind_groups[tier], &[offset]);
        let groups = region.size.map(|s| s.div_ceil(4));
        pass.dispatch_workgroups(groups.x, groups.y, groups.z);
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
