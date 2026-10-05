//! Bake-only caps on the near planes of the two bake cameras.
//!
//! STATUS: the capping does not work on Sponza and should not be trusted. It
//! closes no leak - ceiling hits sit at 102226, exactly the uncapped baseline.
//! What is worth keeping is the machinery, because the stencil pass reuses it:
//! the global texel mask, the set/clear boolean, the tiling quads, and the
//! world-space plane handling. Delete this module once `Heightmap::solid`
//! exists and is written by a stencil pass.
//!
//! Why it fails: capping decides what is solid from surface winding, and the
//! heightmap cannot represent "this column is inside material" at all. It only
//! stores a floor and a ceiling, so a hollow column reads like an ordinary tall
//! interval and probes get placed inside it - which is visible in the debug
//! view. No contour test can fix that, because a vertical ray through a wall
//! column and a vertical ray through an atrium are the same observation: same
//! floor, same ceiling, same normals. It needs a horizontal inside-test.
//!
//! One real finding, in case it saves someone the three dead ends I took:
//! every one of the 27 contours at the split is one-sided, and the single
//! "+1" contour is a genuinely large convex solid - a storey slab. Capping a
//! slab is geometrically correct; the limit is that one split plus one
//! floor/ceiling interval cannot hold a slab at the split *and* the storey
//! above it. That is the multi-frustum plan, not a capping bug.
//!
//! The bake cameras are orthographic along Y, so a wall is edge-on and
//! contributes no texels at all. A column through one reads floor = whatever
//! surface the wall stands on and ceiling = whatever surface it stops at, an
//! interval that runs straight through the wall - which is why the cascade
//! probes inside a wall collect a bilinear share from both storeys and leak.
//!
//! A cap puts a real surface exactly where the clip boundary is, so the column
//! is terminated at the split and reads as solid there instead. Two rules make
//! it work:
//!
//! * The cap goes on the **near plane**, not on `SPLIT`. Everything between the
//!   split and the near plane is clipped, so a cap there would be clipped too,
//!   and a cap at the split would make the upper floor slab's top face visible
//!   - that face is the storey's floor, not its ceiling - collapsing every
//!   upper-storey interval onto the split.
//! * The cap's normal points along the surface's own up direction, so one cap
//!   serves both cameras: the bake clips the far cap away for each of them.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::math::{Vec2, Vec3};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::render::mesh::VertexAttributeValues;

/// Endpoint welding tolerance for chaining cross-section segments into loops,
/// in metres. The two ends of a cut are interpolated along two different edges,
/// so they agree to float precision but not bit-for-bit. This is far below any
/// geometry here and far above that noise.
const WELD: f32 = 1.0e-4;
/// Distance from the plane at which a vertex counts as lying on it. At the
/// scene's scale this separates "on the plane" from "merely near it" without
/// snapping genuinely thin geometry onto the plane.
const ON_PLANE: f32 = 1.0e-6;

// How vertical a surface must be to count as a wall worth capping, as
// |dot(surface_normal, plane_normal)|. Zero is a perfectly vertical wall, one a
// perfectly horizontal floor or ceiling.
//
// The test exists because capping is only meaningful for walls. A wall that
// straddles the split has no cross-section, so a bake camera looking along the
// split sees straight through the gap the wall makes and the column leaks. A
// floor or ceiling straddling the split is not a gap: the camera already sees it
// as a real surface, and capping it too would replace a correct height with a
// synthetic one 2mm away.
const WALL_MAX_TILT: f32 = 0.5;

/// How far the mean of a contour's normals must lean outwards before that
/// contour counts as bounding material.
///
/// For any *convex* cross-section the boundary normals all agree with the
/// loop's outward perpendicular, so a wall modelled with thickness reads close
/// to +1 and a room outline traced off the walls' inward faces reads close to
/// -1. Filling a +1 contour is then correct, and for a wall it is the wall's
/// own cross-section - not a thickness to be guessed at.
///
/// Note this does *not* mean opposing faces cancel. They do not: the two long
/// faces of a wall slab both point outward from the slab, so both score +1.
/// Only a non-convex or inward-facing outline scores below zero.
const SIDE_EPS: f32 = 0.05;

fn key_eq(a: Vec3, b: Vec3) -> bool {
    (a - b).length() <= WELD
}

/// One piece of a cross-section: the edge where a triangle meets the split
/// plane, plus whether the triangle that contributed it is wall-like.
///
/// The flag is carried rather than used to discard the segment. Whether a
/// segment gets capped is decided after the whole contour has been traced,
/// because the contour is what makes it possible to tell a wall run from a
/// stray edge.
#[derive(Clone, Copy)]
struct Seg {
    a: Vec3,
    b: Vec3,
    wall: bool,
    /// Surface normal of the triangle this came from, in the mesh's own space.
    nrm: Vec3,
}

/// What happened to one mesh. Producing no cap is the failure mode that
/// matters: it leaves a column uncapped and the leak returns with nothing in
/// the log to say why, so every skip is counted and reported.
#[derive(Debug, Default, Clone)]
pub struct CapStats {
    /// Meshes whose Y range crossed at least one plane.
    pub crossing: usize,
    /// Cross-section segments cut from the mesh.
    pub segments: usize,
    /// Traced contours whose normals point out of the loop: they bound solid,
    /// and are capped.
    pub contours_solid: usize,
    /// Traced contours whose normals point into the loop: they bound air, a
    /// zero-thickness shell rather than a wall, and filling them would cap the
    /// room instead of the wall.
    pub contours_air: usize,
    /// Closed contours accepted for capping.
    pub runs: usize,
    /// Area of every closed contour, and of the solid ones alone, in square
    /// metres. A single entry in the hundreds means a contour is spanning a
    /// storey rather than a wall.
    pub areas_m2: Vec<f32>,
    pub solid_areas_m2: Vec<f32>,
    /// Runs that were not a whole closed contour, so the cap closes them with a
    /// chord across whatever the non-wall part of the contour left open.
    pub runs_chorded: usize,
    /// Contour segments that no run consumed, because they were not walls.
    pub loops_open: usize,
    /// Points on the cut where more than two segments meet. A watertight mesh
    /// has exactly two at every crossing point, so this means the surface is
    /// non-manifold there and the chain can take a wrong turn.
    pub junctions: usize,
    /// Rings an ear clip could not finish, triangulated short.
    pub loops_short: usize,
    /// Rings capped across a hole in the surface instead of cut cleanly.
    pub loops_chorded: usize,
    /// Triangles emitted across every cap on this mesh.
    pub triangles: usize,
    /// Texels capped, which is what determines the quad count.
    pub quads: usize,
    /// Why this mesh produced nothing, empty if it was cut. Every early return
    /// sets it, so a skipped mesh is never silent.
    pub reject: &'static str,
    /// Widest span the cut planes covered inside this mesh, as a signed distance
    /// either side of the plane. A span that does not straddle zero means the
    /// plane transform missed the mesh, not that the mesh is solid.
    pub span_lo: f32,
    pub span_hi: f32,
    /// Indices pointing past the end of the vertex buffer.
    pub bad_indices: usize,
}

impl CapStats {
    pub fn merge(&mut self, other: &CapStats) {
        self.crossing += other.crossing;
        self.segments += other.segments;
        self.contours_solid += other.contours_solid;
        self.contours_air += other.contours_air;
        self.runs += other.runs;
        self.quads += other.quads;
        self.areas_m2.extend_from_slice(&other.areas_m2);
        self.solid_areas_m2.extend_from_slice(&other.solid_areas_m2);
        self.runs_chorded += other.runs_chorded;
        self.loops_open += other.loops_open;
        self.junctions += other.junctions;
        self.loops_short += other.loops_short;
        self.loops_chorded += other.loops_chorded;
        self.triangles += other.triangles;
        self.bad_indices += other.bad_indices;
        if self.reject.is_empty() {
            self.reject = other.reject;
        }
    }
}

/// Copy `mesh` with a cap added on each world plane in `planes`, or `None` if
/// the mesh crosses none of them or is not a readable triangle list.
///
/// `inverse` is the owning entity's world transform, inverted, so the planes -
/// which are horizontal in world space - can be applied to the mesh's own
/// vertices and the cap still lands back on the plane it was asked for.
///
/// Returns the stats even when no cap could be built: they are the only clue
/// about why a mesh was skipped, so the failure path must not swallow them.
/// The heightmap texel grid, in world space. Caps are rasterised against it
/// rather than triangulated, so it has to be told where the texels are.
#[derive(Clone, Copy)]
pub struct CapGrid {
    /// World position of the corner of texel (0, 0).
    pub min: Vec2,
    /// World metres per texel.
    pub texel: f32,
    pub res: u32,
}

impl CapGrid {
    /// Centre of texel (x, z) in world space.
    fn centre(&self, x: u32, z: u32) -> Vec2 {
        self.min + Vec2::new((x as f32 + 0.5) * self.texel, (z as f32 + 0.5) * self.texel)
    }
}

/// A closed contour traced across the split, already reduced to world 2D and
/// classified. `set` says which side of it is material: true for the side the
/// normals point away from, false for the side they point into.
pub struct Contour {
    pub pts: Vec<Vec2>,
    pub set: bool,
}

/// Cut and classify one mesh, returning its contours and nothing else. The mask
/// is deliberately *not* built here: it has to be built across every mesh, or a
/// wall's outer faces cannot subtract its inner ones.
pub fn cut_contours(mesh: &Mesh, planes: &[f32]) -> (Vec<Contour>, CapStats) {
    let mut reject = "";
    let none = |reject: &'static str| {
        (
            Vec::new(),
            CapStats {
                reject,
                ..Default::default()
            },
        )
    };
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return none("not a triangle list");
    }
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return none("no POSITION");
    };
    // The bake reads the G-buffer normal to decide which way a surface faces,
    // and probe relocation reads it to decide which way to push a probe out.
    // A mesh without normals would cap into a zero normal, which the reduce
    // pass stores as "no surface here" - indistinguishable from a hole.
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        return none("no NORMAL");
    };
    if positions.len() != normals.len() {
        return none("POSITION and NORMAL lengths differ");
    }
    if positions.len() < 3 {
        return none("fewer than 3 vertices");
    }

    let triangles = match mesh.indices() {
        Some(indices) => indices
            .iter()
            .collect::<Vec<_>>()
            .chunks_exact(3)
            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
            .filter(|t| {
                (t[0] as usize) < positions.len()
                    && (t[1] as usize) < positions.len()
                    && (t[2] as usize) < positions.len()
            })
            .collect::<Vec<_>>(),
        None => (0..positions.len() as u32 / 3)
            .map(|i| [3 * i, 3 * i + 1, 3 * i + 2])
            .collect(),
    };
    if triangles.is_empty() {
        return none("no usable triangles");
    }

    let mut stats = CapStats {
        reject,
        ..Default::default()
    };
    reject = "";
    let mut contours: Vec<Contour> = Vec::new();

    // One mask per mesh, cleared as the outer loop and punched back out by the
    // inner one. Only one plane is ever asked for; a second would need its own
    // mask so the two could not cancel each other.
    for &plane_y in planes {
        // The mesh has already been put into world space, so the plane needs no
        // transform and its normal is exact.
        let origin = Vec3::new(0.0, plane_y, 0.0);
        let normal = Vec3::Y;
        let (segments, lo, hi) = cut(&triangles, &positions, &normals, origin, normal);
        stats.span_lo = stats.span_lo.min(lo);
        stats.span_hi = stats.span_hi.max(hi);
        if segments.is_empty() {
            continue;
        }
        stats.crossing += 1;
        stats.segments += segments.len();

        let chained = chain(&segments, origin, normal);
        stats.loops_open += chained.open;
        stats.junctions += chained.junctions;
        // Walk each traced contour and cap only its wall runs.
        //
        // The contour is traced from every crossing segment first and classified
        // afterwards, rather than dropping non-wall segments before the walk.
        // That order is what makes the test meaningful: a contour only closes if
        // the segments that meet at each crossing point are all present, so
        // filtering first leaves every ring open, and an open ring cannot be
        // told apart from a genuine hole in the mesh. Traversed in full, each
        // contour separates into runs of wall and runs of floor, and only the
        // wall runs need closing.
        for ring in &chained.rings {
            // Which side of the contour is solid?
            //
            // Ear clipping a closed contour fills its interior, and that is only
            // the right answer when the contour bounds material. Traced at the
            // split, the same loop means opposite things depending on which way
            // its surfaces face: a wall modelled with thickness crosses as two
            // side faces whose normals point away from each other, and the strip
            // between them is solid, so filling the loop caps the wall. The same
            // wall modelled as a zero-thickness shell crosses as a single loop
            // around the room with every normal pointing the same way, and
            // filling *that* stamps a false ceiling over the whole floor plate -
            // which is what capping the entire scene looks like.
            //
            // So test the normals instead of assuming. Each segment's normal is
            // projected onto the plane and dotted with the contour's outward
            // direction: normals pointing out of the loop mean the loop bounds
            // solid, normals pointing in mean it bounds air. Averaged over the
            // whole contour, so one stray face cannot flip a wall.
            // Project straight to world XZ rather than through `plane_basis`:
            // for a +Y normal that basis comes out as (Z, X), and feeding it to a
            // grid indexed (X, Z) masks every contour transposed - areas still
            // come out right, because a swap preserves them, but the mask lands
            // somewhere else entirely and unrelated contours overlap there.
            let verts: Vec<Vec2> = ring.verts.iter().map(|p| Vec2::new(p.x, p.z)).collect();
            let n = ring.segs.len();
            let mut area = 0.0f32;
            for i in 0..n {
                let a = verts[i];
                let b = verts[i + 1];
                area += a.x * b.y - b.x * a.y;
            }
            if area.abs() < 1.0e-12 {
                continue;
            }
            let outward_sign = if area > 0.0 { -1.0 } else { 1.0 };
            let mut score = 0.0f32;
            for i in 0..n {
                let a = verts[i];
                let b = verts[i + 1];
                let d = b - a;
                let len = d.length();
                if len < 1.0e-12 {
                    continue;
                }
                let out = Vec2::new(d.y / len * outward_sign, -d.x / len * outward_sign);
                let nrm = segments[ring.segs[i]].nrm;
                score += Vec2::new(nrm.x, nrm.z).dot(out);
            }
            let mean = score / n as f32;
            // Cap the side opposite to what the normal faces.
            //
            // Normals pointing out of the loop say the material is inside it, so
            // the loop sets; normals pointing in say the material is outside, so
            // the loop clears. That is a boolean between the two contours a
            // thick wall cuts - the outer faces and the inner faces - and their
            // difference is the wall. Ear clipping cannot express it: it only
            // ever fills a loop's interior, so a contour around a room fills the
            // room. Applying both loops to a texel mask can, and it needs no wall
            // thickness from anyone: the thickness is whatever the geometry says.
            let set = mean > SIDE_EPS;
            if set {
                stats.contours_solid += 1;
            } else {
                stats.contours_air += 1;
            }
            let mut flat = verts;
            if !closed_loop(&flat) {
                stats.runs_chorded += 1;
                continue;
            }
            // Signed area, so a contour that spans a storey is obvious in the
            // log instead of having to be inferred from the texel count.
            let mut area2 = 0.0f32;
            for i in 0..flat.len() {
                let a = flat[i];
                let b = flat[(i + 1) % flat.len()];
                area2 += a.x * b.y - b.x * a.y;
            }
            // `flat` is already in metres, so the shoelace is already square
            // metres. Scaling by the texel again was reporting a 492 m2 contour
            // as 3 m2.
            let area = (area2 * 0.5).abs();
            stats.areas_m2.push(area);
            if set {
                stats.solid_areas_m2.push(area);
            }
            contours.push(Contour { pts: flat, set });
            stats.runs += 1;
        }
    }

    (contours, stats)
}

/// Rasterise every contour into one mask and build the cap mesh from it.
///
/// Global on purpose. A wall's outer faces and inner faces are usually separate
/// meshes, so a per-mesh mask can only ever add: the outer contour sets the
/// whole floor plate and the inner contour punches the room out in a different
/// mask, where it never cancels. One mask for the plane lets them subtract, and
/// the wall thickness is whatever the geometry says it is.
pub fn cap_mesh(
    contours: &[Contour],
    grid: &CapGrid,
    plane_y: f32,
    stats: &mut CapStats,
) -> Option<Mesh> {
    if contours.is_empty() {
        return None;
    }
    let mut mask = vec![false; (grid.res * grid.res) as usize];
    for c in contours {
        rasterise(&c.pts, grid, &mut mask, c.set);
    }

    // One quad per masked texel, tiling exactly.
    //
    // Tiling on texel boundaries matters more than it looks: the seams this
    // replaces came from triangulated polygon edges landing sub-texel away from
    // the mask that selected them, which left a one-texel band of uncapped
    // columns along every cap. Adjacent quads here share their edge exactly, so
    // there is nothing between them for a ray to slip through.
    let cell = grid.texel;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for z in 0..grid.res {
        for x in 0..grid.res {
            if !mask[(z * grid.res + x) as usize] {
                continue;
            }
            let x0 = grid.min.x + x as f32 * cell;
            let z0 = grid.min.y + z as f32 * cell;
            let base = positions.len() as u32;
            for (cx, cz) in [
                (x0, z0),
                (x0 + cell, z0),
                (x0 + cell, z0 + cell),
                (x0, z0 + cell),
            ] {
                positions.push([cx, plane_y, cz]);
                // The bake negates the ceiling normal and keeps the floor one as
                // is, so +Y yields the downward stored normal relocation needs to
                // push a probe out from under a cap.
                normals.push(Vec3::Y.to_array());
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            stats.triangles += 2;
        }
    }
    stats.quads = stats.triangles / 2;
    if positions.is_empty() {
        return None;
    }

    // wgpu silently drops a draw with an out-of-bounds index, which looks
    // exactly like a cap that generated fine but never rendered. Check it here
    // rather than inferring it from an unchanged heightmap.
    let vertex_count = positions.len() as u32;
    let bad = indices.iter().filter(|i| **i >= vertex_count).count();
    if bad > 0 {
        stats.bad_indices += bad;
        return None;
    }

    let mut out = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    out.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    out.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    out.insert_indices(Indices::U32(indices));
    Some(out)
}

/// Cut every triangle that straddles the plane, returning the segments where
/// the plane met the surface.
fn cut(
    triangles: &[[u32; 3]],
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    origin: Vec3,
    normal: Vec3,
) -> (Vec<Seg>, f32, f32) {
    let mut segments: Vec<Seg> = Vec::new();
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    // Against `normal`, the plane's direction in this mesh's own space - not
    // against local Y. The two differ for any rotated mesh, and testing local Y
    // would cap the walls of a mesh lying on its side while ignoring the walls
    // of a mesh rotated about X.
    let is_wall = |t: &[u32; 3]| {
        let n = Vec3::from(normals[t[0] as usize]);
        n.dot(normal).abs() <= WALL_MAX_TILT && n.length_squared() > 0.5
    };
    let face_normal = |t: &[u32; 3]| Vec3::from(normals[t[0] as usize]);
    for t in triangles {
        let p = [
            Vec3::from(positions[t[0] as usize]),
            Vec3::from(positions[t[1] as usize]),
            Vec3::from(positions[t[2] as usize]),
        ];
        let d = [
            (p[0] - origin).dot(normal),
            (p[1] - origin).dot(normal),
            (p[2] - origin).dot(normal),
        ];
        lo = lo.min(d[0]).min(d[1]).min(d[2]);
        hi = hi.max(d[0]).max(d[1]).max(d[2]);
        let above = d.iter().filter(|v| **v > ON_PLANE).count();
        let below = d.iter().filter(|v| **v < -ON_PLANE).count();
        let on = 3 - above - below;
        // A triangle entirely on one side does not reach the plane.
        if above == 0 && below == 0 {
            continue;
        }
        // An edge lying in the plane *is* the cross-section, and dropping it
        // punches a hole in the cap wherever a face is flush with the cut. The
        // third vertex may be on either side: the edge is part of the boundary
        // of the solid regardless.
        if on == 2 {
            let (a, b) = if d[0].abs() <= ON_PLANE && d[1].abs() <= ON_PLANE {
                (p[0], p[1])
            } else if d[0].abs() <= ON_PLANE && d[2].abs() <= ON_PLANE {
                (p[0], p[2])
            } else {
                (p[1], p[2])
            };
            segments.push(Seg {
                a,
                b,
                wall: is_wall(t),
                nrm: face_normal(t),
            });
            continue;
        }
        // A triangle reaching the plane at a single vertex while the other two
        // straddle is a real crossing: the vertex and the far edge's crossing
        // are the two ends of its cross-section. Only a triangle that stays on
        // one side is tangent.
        if above == 0 || below == 0 {
            continue;
        }

        // Collect the crossings. A vertex already on the plane is itself a
        // crossing, which is what keeps a cap continuous across triangles that
        // share a vertex on the plane instead of leaving a notch there.
        let mut hits: Vec<Vec3> = Vec::with_capacity(2);
        for k in 0..3 {
            let (a, b) = (k, (k + 1) % 3);
            if d[a].abs() <= ON_PLANE {
                hits.push(p[a]);
            }
            if (d[a] > ON_PLANE && d[b] < -ON_PLANE) || (d[a] < -ON_PLANE && d[b] > ON_PLANE) {
                let s = d[a] / (d[a] - d[b]);
                hits.push(p[a].lerp(p[b], s));
            }
        }
        if hits.len() >= 2 {
            segments.push(Seg {
                a: hits[0],
                b: hits[1],
                wall: is_wall(t),
                nrm: face_normal(t),
            });
        }
    }
    (segments, lo, hi)
}

/// An orthonormal pair spanning the plane.
fn plane_basis(normal: Vec3) -> (Vec3, Vec3) {
    let helper = if normal.dot(Vec3::Y).abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let u = helper.cross(normal).normalize();
    (u, normal.cross(u))
}

fn plane_2d(p: Vec3, origin: Vec3, normal: Vec3) -> Vec2 {
    let (u, v) = plane_basis(normal);
    let d = p - origin;
    Vec2::new(d.dot(u), d.dot(v))
}

/// The result of chaining, with the failures counted rather than dropped.
/// One traced contour: `segs[i]` is the segment joining `verts[i]` to
/// `verts[i + 1]`, so `verts` always has one more entry than `segs`. For a
/// contour that closed, the last vertex repeats the first.
struct Ring {
    verts: Vec<Vec3>,
    segs: Vec<usize>,
}

struct Chained {
    /// Contours traced across the split plane, closed or not.
    rings: Vec<Ring>,
    /// Segments consumed by no ring.
    open: usize,
    /// Rings capped across a hole in the surface rather than cut cleanly.
    chorded: usize,
    /// Crossing points with more than two segments meeting.
    junctions: usize,
}

/// Chain cut segments end to end into closed rings.
fn chain(segments: &[Seg], origin: Vec3, normal: Vec3) -> Chained {
    let key = |p: &Vec3| -> (i64, i64) {
        let q = plane_2d(*p, origin, normal);
        ((q.x / WELD).round() as i64, (q.y / WELD).round() as i64)
    };
    // Endpoint -> the segments meeting there, flagged with whether that
    // endpoint is the segment's first vertex.
    let mut incident: HashMap<(i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for (i, seg) in segments.iter().enumerate() {
        incident.entry(key(&seg.a)).or_default().push((i, true));
        incident.entry(key(&seg.b)).or_default().push((i, false));
    }

    let junctions = incident.values().filter(|hits| hits.len() > 2).count();
    let mut used = vec![false; segments.len()];
    let mut rings = Vec::new();
    let mut open = 0usize;
    let mut chorded = 0usize;
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        // Vertices of the walk, plus the index of the segment that introduced
        // each one after the first.
        let mut ring = vec![segments[start].a, segments[start].b];
        let mut order = vec![start];
        // Follow the chain forward from the tail until it runs out of unused
        // segments or returns to the head.
        while ring.len() <= segments.len() + 1 {
            let tail = *ring.last().unwrap();
            let next = incident
                .get(&key(&tail))
                .and_then(|hits| hits.iter().find(|(i, _)| !used[*i]))
                .copied();
            let Some((i, at_first)) = next else { break };
            used[i] = true;
            ring.push(if at_first {
                segments[i].b
            } else {
                segments[i].a
            });
            order.push(i);
        }
        // The walk closes by revisiting the head. The repeat is kept, so
        // `verts` stays one longer than `segs` in both the closed and the open
        // case and the consumer can slice a run without special-casing either.
        if ring.len() >= 4 && key(&ring[ring.len() - 1]) == key(&ring[0]) {
            rings.push(Ring {
                verts: ring,
                segs: order,
            });
            continue;
        }
        // A walk that runs out of segments without returning to the head has
        // run into a hole in the surface. Architectural meshes are full of them,
        // and dropping the cross-section leaves exactly the column the cap
        // exists to close, so cap it anyway: ear clipping closes the ring with a
        // chord across the hole. Over-filling a texel marks it solid that was
        // really open, which hides a little light; under-filling leaks.
        if ring.len() >= 3 {
            chorded += 1;
            rings.push(Ring {
                verts: ring,
                segs: order,
            });
        } else {
            open += 1;
        }
    }
    Chained {
        chorded,
        open: open + used.iter().filter(|u| !**u).count(),
        rings,
        junctions,
    }
}

/// Ear clip a simple polygon in the plane. Returns the triangles as indices
/// into `points`, plus how many rings could not be finished.
/// Crossing-number test of whether a point is inside a closed contour.
fn inside_contour(poly: &[Vec2], p: Vec2) -> bool {
    let mut winding = false;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                winding = !winding;
            }
        }
    }
    winding
}

/// Whether a contour closes on itself well enough to bound an area. A contour
/// traced end to end from a closed surface arrives with its last vertex
/// repeating its first; one that hit a hole does not, and filling that would
/// mean inventing the missing edge.
fn closed_loop(pts: &[Vec2]) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let first = pts[0];
    let last = pts[pts.len() - 1];
    (first - last).length() <= 1.0e-4
}

/// Set or clear every texel whose centre falls inside `poly`.
fn rasterise(poly: &[Vec2], grid: &CapGrid, mask: &mut [bool], set: bool) {
    // Only the texels the contour's bounding box touches. A contour the size of
    // a room would otherwise walk the whole 512x512 grid per loop.
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::MIN));
    for p in poly {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let span = grid.texel;
    let fx = ((lo.x - grid.min.x) / span).floor().max(0.0) as u32;
    let fz = ((lo.y - grid.min.y) / span).floor().max(0.0) as u32;
    let tx = ((hi.x - grid.min.x) / span).ceil().max(0.0) as u32;
    let tz = ((hi.y - grid.min.y) / span).ceil().max(0.0) as u32;
    let tx = tx.min(grid.res.saturating_sub(1));
    let tz = tz.min(grid.res.saturating_sub(1));
    for z in fz..=tz {
        for x in fx..=tx {
            let i = (z * grid.res + x) as usize;
            if inside_contour(poly, grid.centre(x, z)) {
                mask[i] = set;
            }
        }
    }
}

fn ear_clip(points: &[Vec2]) -> (Vec<[usize; 3]>, usize) {
    if points.len() < 3 {
        return (Vec::new(), 0);
    }
    let mut order: Vec<usize> = (0..points.len()).collect();
    // Ear clipping wants counter-clockwise order.
    let mut area = 0.0;
    for i in 0..points.len() {
        let (a, b) = (points[order[i]], points[order[(i + 1) % points.len()]]);
        area += a.x * b.y - b.x * a.y;
    }
    if area < 0.0 {
        order.reverse();
    }

    let mut tris = Vec::new();
    // A ring of n vertices needs at most n - 2 ears; the slack absorbs the
    // iterations that find a reflex vertex and move on.
    let mut budget = points.len() * points.len() + 16;
    while order.len() > 3 && budget > 0 {
        budget -= 1;
        let n = order.len();
        let mut clipped = false;
        for k in 0..n {
            let tri = [order[(k + n - 1) % n], order[k], order[(k + 1) % n]];
            let (a, b, c) = (points[tri[0]], points[tri[1]], points[tri[2]]);
            let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if cross <= 0.0 {
                continue;
            }
            if order
                .iter()
                .any(|&j| j != tri[0] && j != tri[1] && j != tri[2] && inside(points[j], a, b, c))
            {
                continue;
            }
            tris.push(tri);
            order.remove(k);
            clipped = true;
            break;
        }
        // Nothing was convex and empty: the ring self-intersects or is
        // collinear. Emit what has been clipped and drop the remainder rather
        // than spinning here.
        if !clipped {
            break;
        }
    }
    if order.len() == 3 {
        tris.push([order[0], order[1], order[2]]);
        (tris, 0)
    } else {
        (tris, 1)
    }
}

/// Point in triangle by consistent sign of the three edge tests.
fn inside(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    let d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
    let d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
    let negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(negative && positive)
}
