//! The horizontal containment signal: a 2D generalized winding number of the
//! scene cut at [`SPLIT`], evaluated once on the CPU and read by `is_solid`.
//!
//! The heightmap is a floor/ceiling map, and a floor/ceiling map has no way to
//! hold a vertical surface. The bake's rays run straight up and down, so they
//! never cross a wall: both sides of a wall record open air, no column is
//! flagged, and the probes lifted into those columns are lit as if they were
//! outdoors. Measured on Sponza, 115 probes sit inside the drawn sphere of a
//! wall and `is_solid` flags none of them.
//!
//! Slicing the mesh at the split plane gives the missing horizontal signal
//! directly from the geometry, with no extra cameras:
//!
//! ```text
//! w(p) = (1/2pi) sum atan2( cross(a-p, b-p), dot(a-p, b-p) )
//! ```
//!
//! over the directed segments `a -> b` produced by cutting every triangle at
//! `y = SPLIT`. Each segment is oriented by `cross(+Y, n)` with `n` the
//! triangle's outward normal, so `w` is ~1 inside material and ~0 in open air.
//! The atan2 form rather than a crossing count matters: Sponza's shells are
//! open, and on open curves the two disagree on 5% of the plane.
//!
//! Built once, when `BakeState` reports ready — which is also when
//! `measure_scene_extent` has seen the mesh set stop changing, so the triangle
//! list is complete.
use bevy::{
    mesh::Indices,
    prelude::*,
    render::mesh::VertexAttributeValues,
};

use crate::heightmap::{Heightmap, MAX_BOUNDS, MIN_BOUNDS, RESOLUTION, SPLIT};
use crate::heightmap_bake::{BakeState, SceneAsset};
use crate::heightmap_view::HeightmapViewMesh;
use crate::CameraProbeObject;

pub struct WindingPlugin;

impl Plugin for WindingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, build_winding_map.after(crate::heightmap_bake::measure_scene_extent));
    }
}

/// One directed segment of the cut, as `(sx, sz, ex, ez)` in world space.
type Segment = [f32; 4];

/// Slices the scene at `SPLIT` and fills `Heightmap::winding`.
///
/// Does nothing until `BakeState` is ready and does nothing afterwards: the
/// map is a property of the mesh, not of the frame.
fn build_winding_map(
    mut done: Local<bool>,
    bake: Res<BakeState>,
    scene: Res<SceneAsset>,
    asset_server: Res<AssetServer>,
    meshes: Res<Assets<Mesh>>,
    query: Query<(&Mesh3d, &GlobalTransform), (Without<HeightmapViewMesh>, Without<CameraProbeObject>)>,
    heightmap: Res<Heightmap>,
    mut images: ResMut<Assets<Image>>,
) {
    if *done || !bake.is_ready() {
        return;
    }
    if !asset_server.is_loaded_with_dependencies(&scene.0) {
        return;
    }

    let Some(mut segments) = slice_scene(&meshes, &query) else {
        // A mesh handle present but not yet in `Assets<Mesh>`: wait rather
        // than latch a partial triangle list.
        return;
    };
    segments.shrink_to_fit();

    let res = RESOLUTION;
    let cell = Vec2::new(
        (MAX_BOUNDS.x - MIN_BOUNDS.x) / res.x as f32,
        (MAX_BOUNDS.y - MIN_BOUNDS.y) / res.y as f32,
    );
    let mut data = vec![0.0f32; (res.x * res.y) as usize];

    // O(texels x segments), so it is threaded over rows. It runs once per
    // process; the alternative, a crossing count per row, is 5x faster and
    // wrong on this asset (see the module docs).
    let row_len = res.x as usize;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = (res.y as usize).div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk) in data.chunks_mut(per * row_len).enumerate() {
            let segments = &segments;
            scope.spawn(move || {
                for (r, row) in chunk.chunks_mut(row_len).enumerate() {
                    let global_row = t * per + r;
                    let gz = MIN_BOUNDS.y + (global_row as f32 + 0.5) * cell.y;
                    for (c, out) in row.iter_mut().enumerate() {
                        let gx = MIN_BOUNDS.x + (c as f32 + 0.5) * cell.x;
                        let mut acc = 0.0f32;
                        for &[sx, sz, ex, ez] in segments.iter() {
                            let ax = sx - gx;
                            let az = sz - gz;
                            let bx = ex - gx;
                            let bz = ez - gz;
                            acc += (ax * bz - az * bx).atan2(ax * bx + az * bz);
                        }
                        *out = acc / (2.0 * std::f32::consts::PI);
                    }
                }
            });
        }
    });

    let Some(mut image) = images.get_mut(&heightmap.winding) else {
        return;
    };
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    image.data = Some(bytes);
    *done = true;
    info!(
        "winding map: {} segments cut at y={:.2}, {}x{} texels",
        segments.len(),
        SPLIT,
        res.x,
        res.y
    );
}

/// World-space directed segments of every scene triangle cut at [`SPLIT`],
/// or `None` if any mesh handle has not resolved yet.
fn slice_scene(
    meshes: &Assets<Mesh>,
    query: &Query<(&Mesh3d, &GlobalTransform), (Without<HeightmapViewMesh>, Without<CameraProbeObject>)>,
) -> Option<Vec<Segment>> {
    let mut segments = Vec::new();
    for (handle, transform) in query {
        let mesh = meshes.get(handle)?;
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };
        // Triangles as index triples; unindexed meshes are already a triple
        // list.
        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U16(i)) => i.iter().map(|&i| i as u32).collect(),
            Some(Indices::U32(i)) => i.clone(),
            None => (0..positions.len() as u32).collect(),
        };
        for tri in indices.as_chunks::<3>().0 {
            let v = [
                transform.transform_point(Vec3::from(positions[tri[0] as usize])),
                transform.transform_point(Vec3::from(positions[tri[1] as usize])),
                transform.transform_point(Vec3::from(positions[tri[2] as usize])),
            ];
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];
            let n = e1.cross(e2);
            if n.length_squared() < 1.0e-16 {
                continue;
            }
            let n = n.normalize();
            let d = [v[0].y - SPLIT, v[1].y - SPLIT, v[2].y - SPLIT];

            // Exactly the two edges that straddle the plane, as in the
            // reference implementation: an edge crosses where `(di >= 0) !=
            // (dj >= 0)`, and a triangle sitting flat on the plane crosses
            // none and is dropped.
            let mut cut = [Vec3::ZERO; 2];
            let mut found = 0;
            for i in 0..3 {
                let j = (i + 1) % 3;
                if (d[i] >= 0.0) == (d[j] >= 0.0) {
                    continue;
                }
                if found == 2 {
                    found = 3;
                    break;
                }
                let t = d[i] / (d[i] - d[j]);
                cut[found] = v[i] + (v[j] - v[i]) * t;
                found += 1;
            }
            if found != 2 {
                continue;
            }

            // Orientation from the plane's own normal crossed with the
            // triangle's: `cross(+Y, n) = (n.z, 0, -n.x)`. The y component of
            // the cut chord is ~0, so only (x, z) matter.
            let dir = Vec2::new(n.z, -n.x);
            let chord = Vec2::new(cut[1].x - cut[0].x, cut[1].z - cut[0].z);
            let (a, b) = if chord.dot(dir) > 0.0 {
                (cut[0], cut[1])
            } else {
                (cut[1], cut[0])
            };
            segments.push([a.x, a.z, b.x, b.z]);
        }
    }
    Some(segments)
}
