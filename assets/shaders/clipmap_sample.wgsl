// Sampling and sphere-tracing the SDF grid (see src/clipmap.rs), shared by
// everything that traces it, so they all agree on bounds and the hit
// threshold: the debug view (sdf_view.wgsl), the 3D cascades and the
// shadow pass.
#define_import_path s_rc::clipmap_sample

// The grid's window at this frame.
struct Clipmap {
    // World voxel coordinates of the window's min corner, in xyz.
    origin: vec4<i32>,
    voxel: f32,
    // Voxels per axis.
    resolution: u32,
}

// Hit threshold, a quarter voxel: trilinear filtering rounds convex edges
// off, so rays grazing past a sharp edge miss it. Measured against Mitsuba
// from outside the doorway, edge misses went 987 (1 cm threshold) -> 211
// (1/6 voxel) -> 15 (1/4), the last ones within 1 cm of the edge, while
// hits stayed within ~2 cm (median) of the true surface. The cost is
// hitting up to that much early; along grazing rays, depth is off by
// threshold / sin(angle), though the hit point stays that close to a real
// surface.
const HIT_FRACTION: f32 = 0.25;
const MIN_STEP: f32 = 0.01;
const MAX_STEPS: u32 = 256u;

fn hit_eps(clipmap: Clipmap) -> f32 {
    return HIT_FRACTION * clipmap.voxel;
}

// Inside the window, with a voxel and a half of margin so trilinear
// filtering never mixes in a texel from the far side of the torus.
fn in_grid(clipmap: Clipmap, p: vec3<f32>) -> bool {
    let lo = (vec3<f32>(clipmap.origin.xyz) + 1.5) * clipmap.voxel;
    let hi = (vec3<f32>(clipmap.origin.xyz) + f32(clipmap.resolution) - 1.5) * clipmap.voxel;
    return all(p > lo) && all(p < hi);
}

// Toroidal: world voxel v's center, (v + 0.5) * voxel, maps to texel
// v mod resolution's center, and repeat addressing does the wrap.
fn grid_uvw(clipmap: Clipmap, p: vec3<f32>) -> vec3<f32> {
    return p / (clipmap.voxel * f32(clipmap.resolution));
}

fn sample_sdf(clipmap: Clipmap, sdf: texture_3d<f32>, s: sampler, p: vec3<f32>) -> f32 {
    return textureSampleLevel(sdf, s, grid_uvw(clipmap, p), 0.0).r;
}

struct March {
    hit: bool,
    t: f32,
    steps: u32,
}

// Sphere-traces [0, max_t] along a unit direction. Stops at the first hit,
// on leaving the grid (a miss), or at max_t.
fn march(
    clipmap: Clipmap,
    sdf: texture_3d<f32>,
    s: sampler,
    origin: vec3<f32>,
    dir: vec3<f32>,
    max_t: f32,
) -> March {
    return march_eps(clipmap, sdf, s, origin, dir, max_t, hit_eps(clipmap));
}

// `march` with an explicit hit threshold.
fn march_eps(
    clipmap: Clipmap,
    sdf: texture_3d<f32>,
    s: sampler,
    origin: vec3<f32>,
    dir: vec3<f32>,
    max_t: f32,
    eps: f32,
) -> March {
    var t = 0.0;
    var steps = 0u;
    for (; steps < MAX_STEPS; steps++) {
        let p = origin + dir * t;
        if !in_grid(clipmap, p) {
            break;
        }
        let d = sample_sdf(clipmap, sdf, s, p);
        if d < eps {
            return March(true, t, steps);
        }
        t += max(d, MIN_STEP);
        if t >= max_t {
            break;
        }
    }
    return March(false, t, steps);
}

// Soft visibility of the sun (a distant disk of cone factor
// k = 1 / tan(angular radius)) from `start` along unit `dir`: Aaltonen's
// estimator (GDC 2018, as in Quilez's soft shadows article), which
// triangulates the closest approach between steps. Validated against
// Mitsuba with sphere lights: one march matched 16 disk shadow rays.
fn sun_visibility(
    clipmap: Clipmap,
    sdf: texture_3d<f32>,
    s: sampler,
    start: vec3<f32>,
    dir: vec3<f32>,
    k: f32,
) -> f32 {
    var res = 1.0;
    // Radius of the previous step's empty sphere, i.e. the step taken.
    var previous = 1e20;
    var t = 0.0;
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = start + dir * t;
        if !in_grid(clipmap, p) {
            break;
        }
        let d = sample_sdf(clipmap, sdf, s, p);
        if d <= 0.0 {
            return 0.0;
        }
        // The triangulation assumes consecutive empty spheres overlap: the
        // step equals the previous sphere's radius, which keeps y <= d.
        let y = min(d * d / (2.0 * previous), d);
        res = min(res, k * sqrt(d * d - y * y) / max(t - y, 1e-4));
        let step = max(d, MIN_STEP);
        previous = step;
        t += step;
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}
