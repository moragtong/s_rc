//! CPU side of the 3D cascades: per-cascade probe windows, buffer offsets,
//! and octahedral direction tables. Must mirror cascades3d.wgsl.

use bevy::math::{Vec2, Vec3, Vec4};

pub const CASCADES: u32 = 4;
/// Cascade-0 probe spacing, m. Doubles per cascade.
pub const SPACING0: f32 = 0.5;
/// Cascade-0 directions: an OCTA_SIDE0^2 octahedral map. The side doubles
/// per cascade (4x directions).
pub const OCTA_SIDE0: u32 = 4;
/// End of cascade 0's interval, m; cascade n covers
/// [INTERVAL0 * 2^(n-1), INTERVAL0 * 2^n] (cascade 0 from 0).
pub const INTERVAL0: f32 = 1.5;
/// Cascade-0 probes exist within this distance of the camera.
pub const RADIUS0: f32 = 15.0;
/// Shading levels: level l has probes every spacing(l) with cascade 0's
/// directions, traces from 0 to INTERVAL0 * 2^l (cascade 0 scaled by 2^l)
/// and merges with cascade l+1, whose interval starts there; it covers
/// RADIUS0 * 2^l. Level 0 is cascade 0. Pixels shade from the finest level
/// covering them, like the clipmap's tiers, so every level has the near
/// field that cascade n > 0 lacks.
pub const LEVELS: u32 = 1;
/// Sub-samples per axis of each cascade-0 direction cell, for exact cosine
/// weighting when shading from 16 directions.
pub const SHADE_SUBSAMPLES: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct CascadeWindow {
    pub spacing: f32,
    pub octa_side: u32,
    /// Ray interval, m.
    pub interval: [f32; 2],
    /// The cascade it merges with (its children); None for the top one.
    pub child: Option<u32>,
    /// Shading levels > 0 only: offset of its rays in the levels buffer.
    pub radiance_base: u32,
    /// Slots per axis (the window is size^3 probes, mapped toroidally).
    pub size: u32,
    /// Probes within this distance of the camera are live.
    pub radius: f32,
    pub tag_base: u32,
    pub hit_base: u32,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub cascades: Vec<CascadeWindow>,
    /// Shading levels 1.. (level 0 is cascades[0]).
    pub levels: Vec<CascadeWindow>,
    /// Rays of all levels > 0, which sizes their radiance buffer.
    pub level_rays: u64,
    /// Largest shading level's children at its directions (the reduce pass).
    pub reduced_rays: u64,
    pub tag_words: u64,
    pub hit_words: u64,
    /// Largest window in rays, which sizes the ping-pong radiance buffers.
    pub max_rays: u64,
}

pub fn spacing(n: u32) -> f32 {
    SPACING0 * (1 << n) as f32
}

pub fn octa_side(n: u32) -> u32 {
    OCTA_SIDE0 << n
}

pub fn interval(n: u32) -> [f32; 2] {
    let end = INTERVAL0 * (1 << n) as f32;
    [if n == 0 { 0.0 } else { end / 2.0 }, end]
}

impl CascadeWindow {
    pub fn rays_per_probe(&self) -> u32 {
        self.octa_side.pow(2)
    }

    /// Segments marched per stored ray: the 3D bilinear fix traces one per
    /// trilinear child; the top cascade has no children.
    pub fn segments(&self) -> u32 {
        if self.child.is_some() { 8 } else { 1 }
    }

    /// u32 words of cached hits per ray (two unorm16 fractions each).
    pub fn hit_words(&self) -> u32 {
        self.segments().div_ceil(2)
    }

    pub fn rays(&self) -> u64 {
        (self.size as u64).pow(3) * self.rays_per_probe() as u64
    }
}

impl Layout {
    pub fn new() -> Self {
        let level_radius = |l: u32| RADIUS0 * (1 << l) as f32;
        // Even, and wide enough that the window centered on the camera's
        // probe holds every probe within `radius`.
        let window = |radius: f32, s: f32| 2 * ((radius / s).ceil() as u32 + 2);
        let (mut tag_base, mut hit_base) = (0u64, 0u64);
        let mut allocate = |w: &mut CascadeWindow| {
            w.tag_base = tag_base as u32;
            w.hit_base = hit_base as u32;
            tag_base += (w.size as u64).pow(3);
            hit_base += w.rays() * w.hit_words() as u64;
        };

        let mut cascades = Vec::new();
        let mut radius = RADIUS0;
        for n in 0..CASCADES {
            let s = spacing(n);
            if n > 0 {
                // Every live parent's 8 trilinear children lie within
                // sqrt(3) child spacings of it; parents are cascade n-1
                // and shading level n-1.
                let parent = if n - 1 < LEVELS { radius.max(level_radius(n - 1)) } else { radius };
                radius = parent + 1.75 * s;
            }
            let mut w = CascadeWindow {
                spacing: s,
                octa_side: octa_side(n),
                interval: interval(n),
                child: (n + 1 < CASCADES).then_some(n + 1),
                radiance_base: 0,
                size: window(radius, s),
                radius,
                tag_base: 0,
                hit_base: 0,
            };
            allocate(&mut w);
            cascades.push(w);
        }

        let mut levels = Vec::new();
        let mut level_rays = 0u64;
        for l in 1..LEVELS {
            let s = spacing(l);
            let mut w = CascadeWindow {
                spacing: s,
                octa_side: OCTA_SIDE0,
                interval: [0.0, INTERVAL0 * (1 << l) as f32],
                child: Some(l + 1),
                radiance_base: level_rays as u32,
                size: window(level_radius(l), s),
                radius: level_radius(l),
                tag_base: 0,
                hit_base: 0,
            };
            allocate(&mut w);
            level_rays += w.rays();
            levels.push(w);
        }
        let max_rays = cascades.iter().map(CascadeWindow::rays).max().unwrap_or(0);
        let reduced_rays = levels
            .iter()
            .map(|w| (cascades[w.child.unwrap() as usize].size as u64).pow(3) * w.rays_per_probe() as u64)
            .max()
            .unwrap_or(0);
        Self { cascades, levels, level_rays, reduced_rays, tag_words: tag_base, hit_words: hit_base, max_rays }
    }
}

/// Octahedral map, [0,1]^2 -> unit sphere; mirrors `oct_decode` in the shader.
pub fn oct_decode(uv: Vec2) -> Vec3 {
    let f = uv * 2.0 - 1.0;
    let mut n = Vec3::new(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = (-n.z).max(0.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize()
}

fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Van Oosterom & Strackee.
    let numerator = a.dot(b.cross(c)).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * numerator.atan2(denominator)
}

/// Solid angle of every cell of a side x side octahedral map, row-major.
/// Octahedral cells aren't equal-area, so merges and shading weight by
/// these. Integrated over a fine sub-grid of spherical triangles.
pub fn octa_solid_angles(side: u32) -> Vec<f32> {
    const SUB: u32 = 16;
    let mut out = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            let mut area = 0.0;
            for sy in 0..SUB {
                for sx in 0..SUB {
                    let corner = |dx: u32, dy: u32| {
                        oct_decode(Vec2::new(
                            (x as f32 + (sx + dx) as f32 / SUB as f32) / side as f32,
                            (y as f32 + (sy + dy) as f32 / SUB as f32) / side as f32,
                        ))
                    };
                    let (a, b, c, d) = (corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1));
                    area += spherical_triangle_area(a, b, c) + spherical_triangle_area(a, c, d);
                }
            }
            out.push(area);
        }
    }
    out
}

/// For shading from cascade 0: per direction cell, SHADE_SUBSAMPLES^2 sub-
/// directions with their solid angles (xyz = direction, w = solid angle),
/// cell-major, so irradiance can weight each cell by the cosine across it
/// rather than at its center.
pub fn shading_subsamples() -> Vec<Vec4> {
    let side = OCTA_SIDE0 * SHADE_SUBSAMPLES;
    let angles = octa_solid_angles(side);
    let mut out = Vec::new();
    for y in 0..OCTA_SIDE0 {
        for x in 0..OCTA_SIDE0 {
            for sy in 0..SHADE_SUBSAMPLES {
                for sx in 0..SHADE_SUBSAMPLES {
                    let (fx, fy) = (x * SHADE_SUBSAMPLES + sx, y * SHADE_SUBSAMPLES + sy);
                    let uv = (Vec2::new(fx as f32, fy as f32) + 0.5) / side as f32;
                    out.push(oct_decode(uv).extend(angles[(fy * side + fx) as usize]));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octahedral_cells_cover_the_sphere() {
        for side in [4, 8, 16] {
            let total: f32 = octa_solid_angles(side).iter().sum();
            assert!((total - 4.0 * std::f32::consts::PI).abs() < 1e-3, "side {side}: {total}");
        }
    }

    #[test]
    fn shading_subsamples_cover_the_sphere() {
        let total: f32 = shading_subsamples().iter().map(|s| s.w).sum();
        assert!((total - 4.0 * std::f32::consts::PI).abs() < 1e-3, "{total}");
        // Cell-major: every sub-sample encodes back into its own cell.
        let per_cell = (SHADE_SUBSAMPLES * SHADE_SUBSAMPLES) as usize;
        for (i, sample) in shading_subsamples().iter().enumerate() {
            let cell = (i / per_cell) as u32;
            let uv = oct_encode(sample.truncate()) * OCTA_SIDE0 as f32;
            let (x, y) = (uv.x.floor() as u32, uv.y.floor() as u32);
            assert_eq!(y * OCTA_SIDE0 + x, cell, "sub-sample {i}");
        }
    }

    /// Inverse of `oct_decode`, for the test above.
    fn oct_encode(d: Vec3) -> Vec2 {
        let n = d / (d.x.abs() + d.y.abs() + d.z.abs());
        let mut p = Vec2::new(n.x, n.y);
        if n.z < 0.0 {
            p = Vec2::new(
                (1.0 - n.y.abs()) * if n.x >= 0.0 { 1.0 } else { -1.0 },
                (1.0 - n.x.abs()) * if n.y >= 0.0 { 1.0 } else { -1.0 },
            );
        }
        p * 0.5 + 0.5
    }
}
