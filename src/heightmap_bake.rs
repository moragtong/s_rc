//! Bakes the heightmap on the GPU from two orthographic cameras.
//!
//! Both sit at the same XZ origin on the `SPLIT` plane, facing opposite ways,
//! so their near planes *are* the split: the down camera's depth test keeps
//! the highest surface below it (the floor) and the up camera's the lowest
//! surface above it (the ceiling). That is what replaces the per-triangle
//! split classification the old CPU rasteriser had to do.
//!
//! The cameras render nothing but prepasses (`RenderTarget::None`), so the
//! only cost is a depth and a deferred G-buffer. One compute pass then reduces
//! both into the map's textures. The structure follows `hi_z.rs`: prepare in
//! `RenderSystems::PrepareBindGroups`, dispatch after `late_deferred_prepass`.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

use bevy::{
    asset::RenderAssetUsages,
    camera::{
        Camera, ImageRenderTarget, OrthographicProjection, Projection, RenderTarget, ScalingMode,
        primitives::MeshAabb, visibility::RenderLayers,
    },
    core_pipeline::{
        Core3dSystems,
        deferred::node::late_deferred_prepass,
        prepass::{DeferredPrepass, DepthPrepass, ViewPrepassTextures},
        schedule::Core3d,
    },
    prelude::*,
    render::mesh::VertexAttributeValues,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::GpuImage,
        view::{ExtractedView, Msaa, RenderVisibleEntities},
    },
};

use crate::bake_caps::{CapGrid, CapStats, Contour, cap_mesh, cut_contours};
use crate::heightmap::{
    BAKE_LAYER, BAKE_NEAR, CAP_INSET, EMPTY_FLOOR, Heightmap, NO_CEILING, RESOLUTION, SCENE_LAYER,
    SPLIT,
};
use crate::heightmap_view::HeightmapViewMesh;

const SHADER_PATH: &str = "shaders/heightmap_bake.wgsl";
const WORKGROUP: u32 = 8;
/// Frames the reduction must run before the bake cameras are switched off.
const BAKE_FRAMES: u32 = 8;
/// Attempts before we give up and leave the map as it is rather than loop.
const MAX_BAKE_ATTEMPTS: u32 = 8;
/// Ceiling texels a bake must fill to count as real. Sponza bakes 102226 of
/// 262144 and a failed attempt fills 0, so this sits in a wide gap between the
/// two and also rejects a mostly-empty partial.
const MIN_CEILING_HITS: u32 = 60_000;

/// Which half of the split a bake camera looks at. Extracted so the render
/// world can tell the two bake views apart from the main camera.
#[derive(Component, Clone, Copy, PartialEq, Eq, ExtractComponent)]
pub enum BakeView {
    Up,
    Down,
}

/// One bake camera's prepass attachments plus its view matrices, gathered by
/// `prepare_bake` so it can pair the two views into a single bind group.
struct ViewBake {
    depth: TextureView,
    gbuffer: TextureView,
    view_from_clip: Mat4,
    world_from_view: Mat4,
}

/// The finished dispatch, rebuilt each frame while the cameras are live.
#[derive(Resource)]
struct BakeDispatch {
    bind_group: BindGroup,
}

/// How far along the one-shot bake is, shared between the two worlds: the
/// cameras need the scene's Y extent to set their far planes, and the
/// reduction needs to know both that and that it hasn't run yet.
#[derive(Resource, Clone, Default)]
pub struct BakeState {
    /// The scene's meshes have spawned and the cameras have their far planes.
    ready: Arc<AtomicBool>,
    /// How many frames the reduction has run on.
    baked: Arc<AtomicU32>,
    /// The bake cameras have been switched off; the outputs are final.
    done: Arc<AtomicBool>,
    /// A readback confirmed the ceiling channel holds a real map, so the bake
    /// can be trusted and a screenshot taken.
    verified: Arc<AtomicBool>,
    /// Bakes started so far, so a scene that never bakes stops retrying.
    attempts: Arc<AtomicU32>,
    /// Set when a bake came back empty and the cameras must come back on.
    retry: Arc<AtomicBool>,
    /// Whether this scene produced cap geometry at all. Without it a scene with
    /// no walls crossing the split would fail the verifier's cap postcondition
    /// forever and retry itself out of attempts.
    caps_expected: Arc<AtomicBool>,
}

impl BakeState {
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// The dispatch is idempotent — it rewrites every texel of every output —
    /// so it is safe to repeat, and repeat we must: nothing orders this node
    /// after the bake cameras' render-graph nodes except `Camera::order`, and a
    /// single dispatch landing on a frame whose prepasses are still uninitialised
    /// would bake nothing and stick, since there is no second attempt.
    pub fn baked_frames(&self) -> u32 {
        self.baked.load(Ordering::Relaxed)
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// Whether the last bake was checked and found to hold real data. Until
    /// this is set the map may be empty, so nothing downstream should treat it
    /// as final.
    pub fn is_verified(&self) -> bool {
        self.verified.load(Ordering::Relaxed)
    }

    /// Ceiling texels a bake must fill to count as real, for the validator.
    pub fn min_ceiling_hits() -> u32 {
        MIN_CEILING_HITS
    }

    /// Whether any cap geometry was spawned, which is what makes the cap
    /// postcondition in the verifier checkable.
    pub fn caps_expected(&self) -> bool {
        self.caps_expected.load(Ordering::Relaxed)
    }

    pub fn set_caps_expected(&self) {
        self.caps_expected.store(true, Ordering::Relaxed);
    }

    pub fn attempts(&self) -> u32 {
        self.attempts.load(Ordering::Relaxed)
    }

    /// Record a bake that produced a usable map.
    pub fn mark_verified(&self) {
        self.verified.store(true, Ordering::Relaxed);
    }

    /// Ask for another attempt, unless we are out of retries.
    pub fn request_retry(&self) -> bool {
        if self.attempts() >= MAX_BAKE_ATTEMPTS {
            return false;
        }
        self.retry.store(true, Ordering::Relaxed);
        true
    }

    fn wants_retry(&self) -> bool {
        self.retry.swap(false, Ordering::Relaxed)
    }

    /// Bring the bake cameras back for another attempt. The reduction rewrites
    /// every texel, so nothing needs clearing first.
    /// Discard a bake that completed against a prepass the scene has since
    /// invalidated - the near-plane caps arriving - and start again. This is not
    /// a failed attempt, so unlike [`Self::rearm`] it does not consume one of the
    /// retries.
    pub fn invalidate(&self) {
        self.baked.store(0, Ordering::Relaxed);
        self.done.store(false, Ordering::Relaxed);
        self.verified.store(false, Ordering::Relaxed);
    }

    fn rearm(&self) {
        self.baked.store(0, Ordering::Relaxed);
        self.done.store(false, Ordering::Relaxed);
        self.verified.store(false, Ordering::Relaxed);
        self.attempts.fetch_add(1, Ordering::Relaxed);
    }

    fn count_bake(&self) -> u32 {
        self.baked.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

/// Render-world handles of the map's output images.
#[derive(Resource, Clone, ExtractResource)]
struct BakeTargets {
    height: Handle<Image>,
    ceiling_normal: Handle<Image>,
    floor_normal: Handle<Image>,
    ceiling_albedo: Handle<Image>,
    floor_albedo: Handle<Image>,
}

/// The scene to measure. Until this has loaded, there are no scene meshes to
/// bake, so the measurement waits on it instead of guessing.
#[derive(Resource, Clone)]
pub struct SceneAsset(pub Handle<bevy::world_serialization::WorldAsset>);

pub struct HeightmapBakePlugin;

impl Plugin for HeightmapBakePlugin {
    fn build(&self, app: &mut App) {
        // Shared by both worlds on purpose. The cameras need the scene's Y
        // extent before they can render anything, but the reduction lives in
        // the render world and a resource inserted in one world doesn't exist
        // in the other, so both have to see this one.
        let state = BakeState::default();
        app.insert_resource(state.clone())
            .init_resource::<OriginalCullModes>()
            .add_plugins((
                ExtractComponentPlugin::<BakeView>::default(),
                ExtractResourcePlugin::<BakeTargets>::default(),
            ))
            .add_systems(Startup, spawn_bake_cameras)
            .add_systems(
                Update,
                (
                    toggle_scene_culling.after(tag_scene_meshes),
                    tag_scene_meshes,
                ),
            )
            .add_systems(
                Update,
                (
                    measure_scene_extent,
                    spawn_bake_caps
                        .after(tag_scene_meshes)
                        .after(measure_scene_extent),
                    finish_bake.after(spawn_bake_caps),
                    rearm_bake.after(finish_bake),
                ),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(state)
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_bake.in_set(RenderSystems::PrepareBindGroups),
            )
            // After the last pass that writes depth and the deferred G-buffer.
            .add_systems(
                Core3d,
                reduce_views
                    .after(late_deferred_prepass)
                    .in_set(Core3dSystems::Prepass),
            );
    }
}

// Bevy keys prepass textures by render target, so the two bake cameras must
// not share one: two `RenderTarget::None { size }` with the same size compare
// equal, so they would alias a single depth/g-buffer pair.
fn bake_target(images: &mut Assets<Image>, label: &'static str) -> ImageRenderTarget {
    let mut image = Image::new_uninit(
        Extent3d {
            width: RESOLUTION.x,
            height: RESOLUTION.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.label = Some(label.into());
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    // Each call adds its own handle, and `ImageRenderTarget` equality compares
    // handles, so the two cameras land on different prepass textures.
    ImageRenderTarget::from(images.add(image))
}

/// Each scene material's cull mode as authored, so it can be put back exactly.
#[derive(Resource, Default)]
struct OriginalCullModes {
    modes: std::collections::HashMap<AssetId<StandardMaterial>, Option<Face>>,
}

/// Turn face culling off while the bake cameras are live, then put it back.
///
/// Bevy 0.19 has no per-camera cull mode: `Camera3d` carries only the depth
/// load op and texture usages, and triangle culling is baked into the
/// specialized pipeline from the material's own `cull_mode` (default
/// `Some(Face::Back)`, see `StandardMaterial::specialize`). The bake cameras
/// render the same materials as the main camera, so there is no way to set this
/// for one view and not the other.
///
/// So it is scoped by lifetime instead: `BakeState::is_done` is set once the
/// bake cameras have been switched off, so the main camera renders every frame
/// it is actually visible with the authored cull modes, and only the bake
/// frames are affected. Turning culling off matters because a single-sided
/// surface whose normals face away from a bake camera would otherwise be
/// culled before it reaches the depth/G-buffer pass, and the recorded
/// floor/ceiling would be some deeper surface instead.
fn toggle_scene_culling(
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<BakeState>,
    mut original: ResMut<OriginalCullModes>,
) {
    if state.is_done() {
        // Restore. Only touch materials that still hold the value we set, so a
        // material re-authored mid-run is not clobbered back to a stale mode.
        for (id, mode) in original.modes.drain() {
            let Some(mut material) = materials.get_mut(id) else {
                continue;
            };
            if material.cull_mode.is_none() {
                material.cull_mode = mode;
            }
        }
        return;
    }

    for (id, material) in materials.iter_mut() {
        // Only write when the value actually changes: touching an asset every
        // frame would emit `AssetEvent::Modified` and churn the pipeline cache.
        if material.cull_mode.is_some() {
            original.modes.insert(id, material.cull_mode);
            material.cull_mode = None;
        }
    }
}

/// Marks a glTF mesh, so the capping pass can find the scene's own meshes
/// without guessing from their layers. Layers are not usable as the filter:
/// `tag_scene_meshes` adds them in the same schedule, and a query keyed on
/// their absence would match or miss depending on which ran first.
#[derive(Component)]
struct SceneMesh;

/// The glTF's meshes default to layer 0 only, which the bake cameras can't
/// see. Add `SCENE_LAYER` and `BAKE_LAYER` to them, without disturbing the
/// debug overlay, which carries its own layer.
///
/// The bake cameras need both the originals and the capped copies: the caps
/// only add a surface on the near planes, they do not replace the geometry
/// behind them. The main camera is on `SCENE_LAYER` only, so it never sees a
/// cap - a cap is a flat horizontal sheet across the middle of the room, not
/// part of the scene.
fn tag_scene_meshes(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<Mesh3d>,
            Without<HeightmapViewMesh>,
            Without<RenderLayers>,
        ),
    >,
) {
    for entity in &query {
        commands.entity(entity).insert((
            RenderLayers::from_layers(&[SCENE_LAYER, BAKE_LAYER]),
            SceneMesh,
        ));
    }
}

/// Spawn a capped copy of every scene mesh on `BAKE_LAYER`.
///
/// The caps sit on the bake cameras' near planes, which is the only place a cap
/// survives clipping: see `bake_caps`. Runs once, after the extent has settled,
/// and the copies outlive a retry, so the cameras simply render them again.
///
/// The copies carry their vertices in world space and sit at the identity
/// transform. Decomposing the source's `GlobalTransform` back into
/// translation/rotation/scale is lossy - a glTF node may hold shear, which TRS
/// cannot express - and a cap built on the lossy form lands somewhere else
/// entirely, where the bake cameras never see it.
///
/// Gated on the scene having finished loading, so the mesh set is complete, and
/// it invalidates any bake that already finished. Both matter: capping early
/// catches only the meshes loaded so far and never runs again, and capping late
/// leaves the bake free to complete against a prepass without the caps - the
/// reduction reads the prepass as it stands, finds a perfectly good *uncapped*
/// heightmap, and the verifier accepts it.
fn spawn_bake_caps(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    heightmap: Res<Heightmap>,
    bake: Res<BakeState>,
    mut cameras: Query<&mut Camera, With<BakeView>>,
    scene: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>, &GlobalTransform), With<SceneMesh>>,
    mut spawned: Local<bool>,
) {
    if *spawned || !bake.is_ready() {
        return;
    }
    // Only the ceiling, and only on the upper plane.
    //
    // The bake cameras sit on `SPLIT` with `near = BAKE_NEAR`, so a cap on the
    // plane just past a camera's near plane is the *nearest* surface that camera
    // sees - which is the whole mechanism, and also why the caps must be one
    // plane and not two. The up camera looks for the ceiling and the leak is
    // there: an upper storey with no ceiling has nothing to stop the ray, so it
    // escapes and the ground-floor texel records a ceiling twelve metres up.
    // The down camera has no such problem - looking for a floor it finds real
    // geometry, or nothing, and "nothing" is already handled by `EMPTY_FLOOR`.
    // Capping it too would make the cap the nearest surface for the down camera
    // and overwrite 76k texels of genuine floor with a value 2mm under the
    // ceiling, collapsing the stored interval to ~0.1m. Every probe then falls
    // under `HEIGHT_TOLERANCE_SPACING` and is discarded, so indirect light
    // disappears entirely - the heightmap still looks plausible, because the
    // numbers are self-consistent, they are just uniformly wrong.
    let planes = [SPLIT + BAKE_NEAR + CAP_INSET];
    let mut stats = CapStats::default();
    let (mut capped_meshes, mut total) = (0usize, 0usize);
    // Every mesh's contours, combined before anything is rasterised.
    let grid = CapGrid {
        min: Vec2::new(heightmap.min_bounds.x, heightmap.min_bounds.y),
        texel: (heightmap.max_bounds.x - heightmap.min_bounds.x) / heightmap.resolution.x as f32,
        res: heightmap.resolution.x,
    };
    let mut contours: Vec<Contour> = Vec::new();
    // Any scene material will do: culling is off during the bake and the cap is
    // only ever drawn into the bake layer's prepass.
    let mut cap_material: Option<MeshMaterial3d<StandardMaterial>> = None;
    for (mesh, material, transform) in &scene {
        total += 1;
        let Some(source) = meshes.get(&mesh.0) else {
            continue;
        };
        let world = into_world_space(source.clone(), transform);
        let (found, mesh_stats) = cut_contours(&world, &planes);
        if !mesh_stats.reject.is_empty() {
            debug!("heightmap cap skipped: {}", mesh_stats.reject);
        }
        if found.is_empty() {
            continue;
        }
        stats.merge(&mesh_stats);
        capped_meshes += 1;
        cap_material.get_or_insert_with(|| material.clone());
        contours.extend(found);
    }

    // One cap proxy for the whole scene. Building it per mesh would put each
    // mesh's contours in a mask of their own, where a wall's inner faces cannot
    // subtract its outer ones and every mask can only add.
    let cap = cap_mesh(&contours, &grid, planes[0], &mut stats);
    if let (Some(cap), Some(material)) = (cap, cap_material) {
        commands.spawn((
            Mesh3d(meshes.add(cap)),
            material.clone(),
            Transform::default(),
            RenderLayers::layer(BAKE_LAYER),
        ));
    } else if !contours.is_empty() {
        warn!("heightmap caps: contours found but no texels masked");
    }
    *spawned = true;
    // Every frame the bake goes on to count must render with the caps present,
    // so throw away whatever it has and bring the cameras back up if they had
    // already switched themselves off.
    if !contours.is_empty() {
        bake.set_caps_expected();
    }
    bake.invalidate();
    for mut camera in &mut cameras {
        camera.is_active = true;
    }
    info!(
        "heightmap caps: {}/{} meshes capped at y={:.3}; \
         {} solid contours, {} air contours skipped, {} left open, \
         {} cap texels, {} triangles, {} non-manifold points; \
         solid contour areas {} m2, all contour areas {} m2",
        capped_meshes,
        total,
        planes[0],
        stats.contours_solid,
        stats.contours_air,
        stats.runs_chorded,
        stats.quads,
        stats.triangles,
        stats.junctions,
        fmt_areas(&stats.solid_areas_m2),
        fmt_areas(&stats.areas_m2)
    );
}

/// Move every vertex of `mesh` through `xform`, returning a new world-space
/// mesh. Normals are rescaled rather than rotated by an inverse transpose, which
/// `Affine3A` does not expose; the scene's capped meshes are uniformly scaled,
/// where the two agree.
/// Largest few areas, biggest first, so the log stays one line.
fn fmt_areas(areas: &[f32]) -> String {
    let mut a: Vec<f32> = areas.to_vec();
    a.sort_by(|x, y| y.partial_cmp(x).unwrap());
    a.iter()
        .take(6)
        .map(|v| format!("{v:.1}"))
        .collect::<Vec<_>>()
        .join("/")
}

fn into_world_space(mut mesh: Mesh, xform: &GlobalTransform) -> Mesh {
    let affine = xform.affine();
    let scale = Vec3::new(
        affine.matrix3.x_axis.length(),
        affine.matrix3.y_axis.length(),
        affine.matrix3.z_axis.length(),
    );
    let safe = scale.max(Vec3::splat(1e-12));
    let inv_scale = Vec3::new(1.0 / safe.x, 1.0 / safe.y, 1.0 / safe.z);
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            *p = affine.transform_point3(Vec3::from(*p)).to_array();
        }
    }
    if let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        for n in normals.iter_mut() {
            *n = affine
                .transform_vector3(Vec3::from(*n) * inv_scale)
                .normalize_or_zero()
                .to_array();
        }
    }
    mesh
}

fn spawn_bake_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    heightmap: Res<Heightmap>,
) {
    let spawn = |commands: &mut Commands, images: &mut Assets<Image>, view: BakeView| {
        let direction = match view {
            BakeView::Up => 1.0,
            BakeView::Down => -1.0,
        };
        let label = match view {
            BakeView::Up => "bake_up_target",
            BakeView::Down => "bake_down_target",
        };
        // Both cameras share the same XZ origin (0, *, 0) and the same
        // near plane; only the facing differs, which is what splits the scene
        // into ceiling and floor without any per-triangle classification.
        commands.spawn((
            Camera3d::default(),
            Camera {
                // Below the main camera's 0, so both bake cameras render their
                // prepasses before the node the reduction runs in.
                order: match view {
                    BakeView::Up => -2,
                    BakeView::Down => -1,
                },
                ..default()
            },
            Projection::Orthographic(OrthographicProjection {
                // The near plane sits just off the split, so anything on the
                // far side of it is clipped away. The caps go here too, just
                // inside it: see `bake_caps`.
                near: BAKE_NEAR,
                // Patched by measure_scene_extent once the scene has loaded.
                far: 1.0,
                scaling_mode: ScalingMode::Fixed {
                    width: heightmap.max_bounds.x - heightmap.min_bounds.x,
                    height: heightmap.max_bounds.y - heightmap.min_bounds.y,
                },
                ..OrthographicProjection::default_2d()
            }),
            // Bevy keys each view's prepass textures by its render target, so
            // two `RenderTarget::None { size }` of the same size compare equal
            // and would share one depth/G-buffer pair, with the second camera
            // clearing over the first. Give each its own image instead.
            RenderTarget::Image(bake_target(images, label)),
            // Only `BAKE_LAYER`, which the directional light is not on: a
            // light these cameras can see makes Bevy build shadow cascades for
            // views that have no shadow target, which panics light_pbr.
            RenderLayers::layer(BAKE_LAYER),
            Msaa::Off,
            DepthPrepass,
            DeferredPrepass,
            Transform::from_xyz(0.0, SPLIT, 0.0)
                .looking_at(Vec3::new(0.0, SPLIT + direction, 0.0), Vec3::NEG_Z),
            view,
        ));
    };

    spawn(&mut commands, &mut images, BakeView::Up);
    spawn(&mut commands, &mut images, BakeView::Down);

    // "min"/"max" name the two channels of the height texture, not low and
    // high: surface_light.wgsl pairs h.x (ceiling) with *_min and h.y (floor)
    // with *_max. Swapping these silently inverts every normal and leaves the
    // sun permanently shadowed.
    commands.insert_resource(BakeTargets {
        height: heightmap.texture.clone(),
        ceiling_normal: heightmap.metadata_min.clone(),
        floor_normal: heightmap.metadata_max.clone(),
        ceiling_albedo: heightmap.albedo_min.clone(),
        floor_albedo: heightmap.albedo_max.clone(),
    });
}

/// Sets each camera's far plane from the scene's Y extent, and `top` from the
/// scene's max: nothing the bake can see is above it.
fn measure_scene_extent(
    mut settle: Local<Option<(usize, u32, u32)>>,
    bake: Res<BakeState>,
    scene: Res<SceneAsset>,
    asset_server: Res<AssetServer>,
    meshes: Res<Assets<Mesh>>,
    query: Query<(&Mesh3d, &GlobalTransform), Without<HeightmapViewMesh>>,
    mut cameras: Query<(&mut Projection, &BakeView)>,
    mut heightmap: ResMut<Heightmap>,
) {
    if bake.is_ready() {
        return;
    }
    // Wait for the real scene rather than a mesh count: the glTF and its
    // dependencies can take several frames, and until then every Mesh3d in
    // the world is one of our own debug planes.
    if !asset_server.is_loaded_with_dependencies(&scene.0) {
        return;
    }

    // Meshes' local AABBs, transformed into world space.
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut found = false;
    for (handle, transform) in &query {
        let Some(mesh) = meshes.get(handle) else {
            continue;
        };
        let Some(aabb) = mesh.compute_aabb() else {
            continue;
        };
        let lo = transform.transform_point(Vec3::from(aabb.min())).y;
        let hi = transform.transform_point(Vec3::from(aabb.max())).y;
        min_y = min_y.min(lo);
        max_y = max_y.max(hi);
        found = true;
    }
    if !found {
        // Loaded but nothing spawned yet; keep waiting.
        return;
    }

    // The asset reports loaded before WorldAsset has finished spawning its
    // entities, so one stable-looking frame proves nothing: measuring a partial
    // scene yields a short max_y, which clips the bake cameras' far planes (no
    // ceiling/floor captured) and shrinks `top` (rays escape to sky early, which
    // reads as light leaking). Latch only once the mesh set has stopped
    // changing, and measure both extents from all eight AABB corners since the
    // scene root may carry a rotation.
    let signature = (query.iter().count(), min_y.to_bits(), max_y.to_bits());
    let settled = settle.as_ref() == Some(&signature);
    *settle = Some(signature);
    if !settled {
        return;
    }

    info!(
        "heightmap bake scene loaded: {} meshes, y {:.2}..{:.2}, \
         {}x{} texels over {:.2}x{:.2} m ({:.3} m per texel)",
        query.iter().count(),
        min_y,
        max_y,
        heightmap.resolution.x,
        heightmap.resolution.y,
        heightmap.max_bounds.x - heightmap.min_bounds.x,
        heightmap.max_bounds.y - heightmap.min_bounds.y,
        (heightmap.max_bounds.x - heightmap.min_bounds.x) / heightmap.resolution.x as f32,
    );

    for (mut projection, view) in &mut cameras {
        let span = match view {
            BakeView::Up => max_y - SPLIT,
            BakeView::Down => SPLIT - min_y,
        };
        if let Projection::Orthographic(o) = &mut *projection {
            o.far = span.max(0.1);
        }
    }
    heightmap.top = max_y;
    bake.ready.store(true, Ordering::Relaxed);
    info!(
        "heightmap bake: {} meshes, y in [{:.2}, {:.2}]",
        query.iter().count(),
        min_y,
        max_y
    );
}

/// Once the cameras have rendered their prepasses, drop the dispatch and
/// Brings the bake cameras back after a verified-empty attempt.
///
/// The bake is bimodal: a completed attempt fills 111200 of 262144 ceiling
/// texels, a failed one fills none, with nothing in between and both states
/// perfectly repeatable. Which of the two happens is decided by when the
/// prepasses land relative to the reduction, so re-running is a genuine retry
/// rather than a repeat of the same frame.
fn rearm_bake(bake: Res<BakeState>, mut cameras: Query<&mut Camera, With<BakeView>>) {
    if !bake.wants_retry() {
        return;
    }
    for mut camera in &mut cameras {
        camera.is_active = true;
    }
    info!(
        "heightmap bake attempt {} was empty; retrying",
        bake.attempts() + 1
    );
    bake.rearm();
}

fn finish_bake(bake: Res<BakeState>, mut cameras: Query<&mut Camera, With<BakeView>>) {
    // Wait for the dispatch to have run several times, not a fixed number of
    // frames after the scene loaded: the compute pipeline takes an arbitrary
    // number of frames to compile, and the prepasses only exist while these
    // cameras are live, so switching them off early would bake nothing.
    // Several successful frames also means the last dispatch read real prepasses
    // rather than a freshly allocated, not-yet-rendered texture.
    if bake.baked_frames() < BAKE_FRAMES {
        return;
    }
    // Only the bake cameras: the main camera matches a bare `&mut Camera` too,
    // and deactivating it leaves a black window.
    //
    // Deactivate rather than remove `Camera`: `extract_cascades` keys off
    // `Camera::is_active`, so removing it strands a camera that still has a
    // `Projection` and `GlobalTransform` but never gets shadow cascades
    // built, which panics light_pbr when the light is visible to it.
    for mut camera in &mut cameras {
        camera.is_active = false;
    }
    // Latch only after the cameras are off. Deactivating them drops
    // `ViewPrepassTextures`, and Bevy's prepass textures are pooled: a later
    // dispatch holding the stale `BakeDispatch` would read whatever the pool
    // handed that texture to next (the main camera's prepasses), overwriting a
    // good map with an empty one.
    bake.finish();
}

#[derive(Resource)]
struct BakePipeline {
    layout: BindGroupLayoutDescriptor,
    // Holding the handle is load-bearing, not incidental: it's the last strong
    // reference to the shader asset, and dropping it unloads the shader out
    // from under the cached pipeline.
    #[allow(dead_code)]
    shader: Handle<Shader>,
    pipeline: CachedComputePipelineId,
}

/// Must match the `View`/`Uniforms` structs in `heightmap_bake.wgsl`.
#[derive(ShaderType, Clone)]
struct BakeUniforms {
    up_view_from_clip: Mat4,
    up_world_from_view: Mat4,
    down_view_from_clip: Mat4,
    down_world_from_view: Mat4,
    hmap_res: UVec2,
    empty_ceiling: f32,
    empty_floor: f32,
    up_flip_x: u32,
    down_flip_x: u32,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "heightmap_bake_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_depth_2d(),
                texture_2d(TextureSampleType::Uint),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Uint),
                uniform_buffer::<BakeUniforms>(false),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("heightmap_bake".into()),
        layout: vec![layout.clone()],
        shader: shader.clone(),
        entry_point: Some("bake".into()),
        ..default()
    });
    commands.insert_resource(BakePipeline {
        layout,
        shader,
        pipeline,
    });
}

fn prepare_bake(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<BakePipeline>,
    heightmap: Res<Heightmap>,
    targets: Res<BakeTargets>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    views: Query<(
        &BakeView,
        &ViewPrepassTextures,
        &ExtractedView,
        &RenderVisibleEntities,
    )>,
) {
    // Clear first, unconditionally. Every early return below would otherwise
    // leave the previous frame's `BakeDispatch` in the resource set, and
    // `reduce_views` cannot tell a stale one from a fresh one - it would go on
    // to count BAKE_FRAMES bakes against one bind group, reading whatever that
    // group still points at.
    //
    // This is not hypothetical. `spawn_bake_caps` calls `invalidate`, which
    // switches the bake cameras back on; `ViewPrepassTextures` are absent for a
    // frame or two while the render graph rebuilds, and in that window
    // `prepare_bake` bails out early. With the old dispatch left behind, the
    // bake then "completed" against prepass textures filled *before* the caps
    // existed and wrote a map identical to the uncapped one - which looked like
    // the caps having had no effect, and came and went between runs depending on
    // how the frames landed. Clearing up front is the only way to be sure every
    // early path is covered.
    commands.remove_resource::<BakeDispatch>();

    let mut up: Option<ViewBake> = None;
    let mut down: Option<ViewBake> = None;

    for (view, prepass, extracted_view, _visible) in &views {
        let (Some(depth), Some(gbuffer)) = (&prepass.depth, &prepass.deferred) else {
            // The view saw no opaque geometry, so it got no prepass phase and
            // no textures; nothing to reduce.
            continue;
        };
        if depth.texture.texture.sample_count() > 1 {
            warn_once!("heightmap bake requires Msaa::Off; skipping frame");
            return;
        }
        let bake = ViewBake {
            depth: depth.texture.default_view.clone(),
            gbuffer: gbuffer.texture.default_view.clone(),
            // world = world_from_view * (view_from_clip * clip).xyz / w
            view_from_clip: extracted_view.clip_from_view.inverse(),
            world_from_view: extracted_view.world_from_view.to_matrix(),
        };
        match view {
            BakeView::Up => up = Some(bake),
            BakeView::Down => down = Some(bake),
        }
    }

    let (Some(up), Some(down)) = (up, down) else {
        return;
    };

    let image = |handle: &Handle<Image>| gpu_images.get(handle).map(|i| i.texture_view.clone());
    let (
        Some(height),
        Some(ceiling_normal),
        Some(floor_normal),
        Some(ceiling_albedo),
        Some(floor_albedo),
    ) = (
        image(&targets.height),
        image(&targets.ceiling_normal),
        image(&targets.floor_normal),
        image(&targets.ceiling_albedo),
        image(&targets.floor_albedo),
    )
    else {
        // Not uploaded to the GPU yet.
        return;
    };

    let uniforms = BakeUniforms {
        up_view_from_clip: up.view_from_clip,
        up_world_from_view: up.world_from_view,
        down_view_from_clip: down.view_from_clip,
        down_world_from_view: down.world_from_view,
        hmap_res: heightmap.resolution,
        empty_ceiling: NO_CEILING,
        empty_floor: EMPTY_FLOOR,
        // Looking along +Y is a mirror in x for a right-handed basis, so the
        // up view's texels run the other way across the map.
        up_flip_x: 1,
        down_flip_x: 0,
    };
    let mut bytes = encase::UniformBuffer::new(Vec::new());
    bytes.write(&uniforms).unwrap();
    let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("heightmap_bake_uniform"),
        contents: bytes.as_ref(),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    let bind_group = render_device.create_bind_group(
        "heightmap_bake",
        &layout,
        &BindGroupEntries::sequential((
            &up.depth,
            &up.gbuffer,
            &down.depth,
            &down.gbuffer,
            uniform.as_entire_binding(),
            &height,
            &ceiling_normal,
            &floor_normal,
            &ceiling_albedo,
            &floor_albedo,
        )),
    );
    commands.insert_resource(BakeDispatch { bind_group });
}

fn reduce_views(
    dispatch: Option<Res<BakeDispatch>>,
    views: ViewQuery<(Option<&BakeView>, &ExtractedView)>,
    bake: Res<BakeState>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<BakePipeline>,
    heightmap: Res<Heightmap>,
    mut ctx: RenderContext,
) {
    // Every camera gets its own run of the Core3d schedule, so this runs once
    // per view. The bake cameras are ordered below the main camera's 0, so
    // their nodes run first: the main camera's node is the first point at
    // which *both* of their prepasses have been filled.
    if views.into_inner().0.is_some() {
        return;
    }
    // Nothing to reduce until the scene has spawned and the cameras have their
    // far planes.
    if !bake.is_ready() {
        return;
    }
    // Once the bake cameras are off their prepasses are gone and pooled, so a
    // dispatch now would read recycled memory and destroy a finished map.
    if bake.is_done() {
        return;
    }
    let (Some(dispatch), Some(pipeline)) = (
        dispatch,
        pipeline_cache.get_compute_pipeline(pipelines.pipeline),
    ) else {
        // The compute pipeline needs a few frames to compile; the cameras stay
        // live until `finish_bake` has seen enough baked frames.
        return;
    };

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("heightmap_bake"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &dispatch.bind_group, &[]);
    let res = heightmap.resolution;
    pass.dispatch_workgroups(res.x.div_ceil(WORKGROUP), res.y.div_ceil(WORKGROUP), 1);

    bake.count_bake();
}
