//! Per-mesh SDF bake on top of parry3d (Lumen's "mesh distance fields"):
//! each mesh gets its own grid in mesh-local space, baked on the CPU and
//! cached to disk; the clipmap composes instances of these grids on the GPU.
//!
//! Sign comes from parry's pseudo-normal inside test, which assumes closed,
//! consistently wound meshes. That holds for the analytic room; open meshes
//! (Sponza has some) will need a ray-voting sign test on top, not done yet.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use bevy::math::{UVec3, Vec3};
use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
use parry3d::math::Vector as PVec3;
use parry3d::query::PointQuery;
use parry3d::shape::{TriMesh, TriMeshFlags};

/// A signed distance grid. Value `(x, y, z)` is the distance at the center
/// of that voxel: `min + (index + 0.5) * voxel`. Negative inside.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfGrid {
    pub min: Vec3,
    pub voxel: f32,
    pub dims: UVec3,
    pub values: Vec<f32>,
}

impl SdfGrid {
    #[cfg(test)]
    pub fn index(&self, cell: UVec3) -> usize {
        (cell.x + self.dims.x * (cell.y + self.dims.y * cell.z)) as usize
    }

    pub fn cell_center(&self, cell: UVec3) -> Vec3 {
        self.min + (cell.as_vec3() + 0.5) * self.voxel
    }
}

/// Positions and triangle indices of a Bevy mesh, in mesh-local space.
pub fn mesh_triangles(mesh: &Mesh) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
    let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        return Err("mesh has no Float32x3 positions".into());
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U16(i)) => i.iter().map(|&v| v as u32).collect(),
        Some(Indices::U32(i)) => i.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    Ok((positions.clone(), indices))
}

/// Bakes a grid covering the mesh's bounds plus `padding` voxels each side.
pub fn bake(positions: &[[f32; 3]], indices: &[u32], voxel: f32, padding: u32) -> Result<SdfGrid, String> {
    let vertices: Vec<PVec3> = positions.iter().map(|p| PVec3::new(p[0], p[1], p[2])).collect();
    let triangles: Vec<[u32; 3]> = indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    // Bevy meshes duplicate vertices per face (for flat normals); merging
    // them restores the connectivity the pseudo-normals need.
    let mesh = TriMesh::with_flags(
        vertices,
        triangles,
        TriMeshFlags::ORIENTED | TriMeshFlags::MERGE_DUPLICATE_VERTICES,
    )
    .map_err(|e| format!("{e:?}"))?;

    let (lo, hi) = positions.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(Vec3::from(*p)), hi.max(Vec3::from(*p))),
    );
    let min = lo - Vec3::splat(padding as f32 * voxel);
    let dims = ((hi - lo) / voxel).ceil().as_uvec3().max(UVec3::ONE) + UVec3::splat(2 * padding);
    let mut grid = SdfGrid { min, voxel, dims, values: vec![0.0; (dims.x * dims.y * dims.z) as usize] };

    // Parallel over z-slabs; parry queries are read-only.
    let slice = (dims.x * dims.y) as usize;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let slabs_per_thread = (dims.z as usize).div_ceil(threads);
    let grid_ref = &grid;
    let chunks: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let mesh = &mesh;
                scope.spawn(move || {
                    // Grids thinner in z than the thread count leave the last
                    // threads with nothing: clamp so their (empty) range still
                    // starts inside the grid.
                    let z1 = ((t + 1) * slabs_per_thread).min(dims.z as usize);
                    let z0 = (t * slabs_per_thread).min(z1);
                    let mut out = Vec::with_capacity(z1.saturating_sub(z0) * slice);
                    for z in z0..z1 {
                        for y in 0..dims.y {
                            for x in 0..dims.x {
                                let p = grid_ref.cell_center(UVec3::new(x, y, z as u32));
                                let projection = mesh.project_local_point(PVec3::new(p.x, p.y, p.z), false);
                                let q = projection.point;
                                let distance = Vec3::new(q.x, q.y, q.z).distance(p);
                                out.push(if projection.is_inside { -distance } else { distance });
                            }
                        }
                    }
                    (z0 * slice, out)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (start, values) in chunks {
        grid.values[start..start + values.len()].copy_from_slice(&values);
    }
    Ok(grid)
}

const CACHE_MAGIC: &[u8; 4] = b"SDF1";

/// `bake`, cached in `cache_dir` by a hash of the inputs.
pub fn bake_cached(
    positions: &[[f32; 3]],
    indices: &[u32],
    voxel: f32,
    padding: u32,
    cache_dir: &Path,
) -> Result<SdfGrid, String> {
    let mut hasher = DefaultHasher::new();
    CACHE_MAGIC.hash(&mut hasher);
    for p in positions {
        p.map(f32::to_bits).hash(&mut hasher);
    }
    indices.hash(&mut hasher);
    (voxel.to_bits(), padding).hash(&mut hasher);
    let path: PathBuf = cache_dir.join(format!("{:016x}.sdf", hasher.finish()));

    if let Ok(grid) = read_grid(&path) {
        return Ok(grid);
    }
    let grid = bake(positions, indices, voxel, padding)?;
    std::fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
    write_grid(&path, &grid).map_err(|e| e.to_string())?;
    Ok(grid)
}

fn write_grid(path: &Path, grid: &SdfGrid) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(32 + grid.values.len() * 4);
    bytes.extend_from_slice(CACHE_MAGIC);
    for v in [grid.dims.x, grid.dims.y, grid.dims.z] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in [grid.min.x, grid.min.y, grid.min.z, grid.voxel] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in &grid.values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(path)?.write_all(&bytes)
}

fn read_grid(path: &Path) -> Result<SdfGrid, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut bytes)).map_err(|e| e.to_string())?;
    if bytes.len() < 32 || &bytes[..4] != CACHE_MAGIC {
        return Err("bad cache file".into());
    }
    let word = |i: usize| <[u8; 4]>::try_from(&bytes[4 + i * 4..8 + i * 4]).unwrap();
    let dims = UVec3::new(u32::from_le_bytes(word(0)), u32::from_le_bytes(word(1)), u32::from_le_bytes(word(2)));
    let f = |i: usize| f32::from_le_bytes(word(i));
    let count = (dims.x * dims.y * dims.z) as usize;
    if bytes.len() != 32 + count * 4 {
        return Err("truncated cache file".into());
    }
    let values = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Ok(SdfGrid { min: Vec3::new(f(3), f(4), f(5)), voxel: f(6), dims, values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene3d::SceneBox;
    use bevy::prelude::{Cuboid, MeshBuilder, Meshable};

    /// A baked Bevy cuboid mesh must match the exact box SDF at every voxel,
    /// inside and out: this exercises the Bevy -> parry path, the vertex
    /// merge and the pseudo-normal sign.
    #[test]
    fn baked_cuboid_matches_analytic_sdf() {
        // Thin in y; and thin in z: 9 z-slabs, which on 8 threads left the last
        // thread starting past the end of the grid before the clamp.
        for half in [Vec3::new(1.3, 0.1, 0.7), Vec3::new(1.3, 0.7, 0.08)] {
            check_cuboid(half);
        }
    }

    fn check_cuboid(half: Vec3) {
        let (positions, indices) = mesh_triangles(&Cuboid::from_size(half * 2.0).mesh().build()).unwrap();
        let grid = bake(&positions, &indices, 0.0625, 3).unwrap();
        let exact = SceneBox { center: Vec3::ZERO, half_size: half, albedo: Vec3::ONE };

        let (mut max_error, mut inside) = (0.0f32, 0);
        for z in 0..grid.dims.z {
            for y in 0..grid.dims.y {
                for x in 0..grid.dims.x {
                    let cell = UVec3::new(x, y, z);
                    let expected = exact.sdf(grid.cell_center(cell));
                    max_error = max_error.max((grid.values[grid.index(cell)] - expected).abs());
                    inside += (expected < 0.0) as u32;
                }
            }
        }
        assert!(inside > 0, "grid should contain interior voxels");
        assert!(max_error < 1e-4, "max |baked - exact| = {max_error}");
    }

    #[test]
    fn cache_round_trips() {
        let (positions, indices) = mesh_triangles(&Cuboid::new(1.0, 2.0, 0.5).mesh().build()).unwrap();
        let dir = std::env::temp_dir().join(format!("s_rc_sdf_test_{}", std::process::id()));
        let baked = bake_cached(&positions, &indices, 0.125, 2, &dir).unwrap();
        let cached = bake_cached(&positions, &indices, 0.125, 2, &dir).unwrap();
        assert_eq!(baked, cached);
        std::fs::remove_dir_all(dir).ok();
    }
}
