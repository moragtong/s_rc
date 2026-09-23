// Sampling and sphere-tracing the SDF clipmap (see src/clipmap.rs), shared
// by everything that traces it, so they all agree on tiers and hit
// thresholds: the debug view (sdf_view.wgsl) and the 3D cascades.
#define_import_path s_rc::clipmap_sample

// The clipmap's tier windows at this frame.
struct Clipmap {
    // Each tier's window origin (world voxel coordinates) in xyz.
    origins: array<vec4<i32>, 3>,
    voxels: vec4<f32>,
    // Voxels per axis of each tier's window.
    resolutions: vec4<u32>,
}

// Hit thresholds are a fraction of each tier's voxel, erring toward
// over-occlusion (lost energy) rather than leaks (ARCHITECTURE.md):
// - Coarse tiers, half a voxel: a wall thinner than the voxel sits between
//   voxel centers, so the stored distances stay positive on both sides; the
//   smallest interpolated value across it is at most (voxel - thickness) / 2.
//   A fixed small threshold walks straight through it.
// - Tier 0, a quarter voxel: trilinear filtering rounds convex edges off, so
//   rays grazing past a sharp edge miss it. Measured against Mitsuba from
//   outside the doorway, edge misses went 987 (1 cm threshold) -> 211 (1/6
//   voxel) -> 15 (1/4), the last ones within 1 cm of the edge, while hits
//   stayed within ~2 cm (median) of the true surface.
// The cost is hitting up to that much early; along grazing rays, depth is
// off by threshold / sin(angle), though the hit point stays that close to a
// real surface.
const TIER0_HIT_FRACTION: f32 = 0.25;
const MIN_STEP: f32 = 0.01;
const MAX_STEPS: u32 = 256u;
const OUTSIDE: u32 = 3u;

fn hit_eps(clipmap: Clipmap, t: u32) -> f32 {
    return select(TIER0_HIT_FRACTION, 0.5, t > 0u) * clipmap.voxels[t];
}

// Inside tier t's window, with a voxel and a half of margin so trilinear
// filtering never mixes in a texel from the far side of the torus.
fn in_tier(clipmap: Clipmap, p: vec3<f32>, t: u32) -> bool {
    let voxel = clipmap.voxels[t];
    let origin = vec3<f32>(clipmap.origins[t].xyz);
    let lo = (origin + 1.5) * voxel;
    let hi = (origin + f32(clipmap.resolutions[t]) - 1.5) * voxel;
    return all(p > lo) && all(p < hi);
}

// Finest tier containing p; OUTSIDE = beyond the clipmap.
fn tier_at(clipmap: Clipmap, p: vec3<f32>) -> u32 {
    for (var t = 0u; t < 3u; t++) {
        if in_tier(clipmap, p, t) {
            return t;
        }
    }
    return OUTSIDE;
}

// Toroidal: world voxel v's center, (v + 0.5) * voxel, maps to texel
// v mod resolution's center, and repeat addressing does the wrap.
fn sample_tier(
    clipmap: Clipmap,
    tier0: texture_3d<f32>,
    tier1: texture_3d<f32>,
    tier2: texture_3d<f32>,
    s: sampler,
    p: vec3<f32>,
    t: u32,
) -> f32 {
    let uvw = p / (clipmap.voxels[t] * f32(clipmap.resolutions[t]));
    switch t {
        case 0u: { return textureSampleLevel(tier0, s, uvw, 0.0).r; }
        case 1u: { return textureSampleLevel(tier1, s, uvw, 0.0).r; }
        default: { return textureSampleLevel(tier2, s, uvw, 0.0).r; }
    }
}

struct March {
    hit: bool,
    t: f32,
    tier: u32,
    steps: u32,
}

// Sphere-traces [0, max_t] along a unit direction. Stops at the first hit,
// on leaving the clipmap, or at max_t.
fn march(
    clipmap: Clipmap,
    tier0: texture_3d<f32>,
    tier1: texture_3d<f32>,
    tier2: texture_3d<f32>,
    s: sampler,
    origin: vec3<f32>,
    dir: vec3<f32>,
    max_t: f32,
) -> March {
    var t = 0.0;
    var tier = OUTSIDE;
    var steps = 0u;
    for (; steps < MAX_STEPS; steps++) {
        let p = origin + dir * t;
        tier = tier_at(clipmap, p);
        if tier == OUTSIDE {
            break;
        }
        let d = sample_tier(clipmap, tier0, tier1, tier2, s, p, tier);
        if d < hit_eps(clipmap, tier) {
            return March(true, t, tier, steps);
        }
        t += max(d, MIN_STEP);
        if t >= max_t {
            break;
        }
    }
    return March(false, t, tier, steps);
}

// Soft visibility of the sun (a distant disk of cone factor
// k = 1 / tan(angular radius)) from `start` along unit `dir`: Aaltonen's
// estimator (GDC 2018, as in Quilez's soft shadows article), which
// triangulates the closest approach between steps. Validated against
// Mitsuba with sphere lights (sdf_view.wgsl's history): one march matched
// 16 disk shadow rays. In coarse tiers, distances are to the surface
// dilated by the tier's hit threshold, the conservative surface every hard
// hit uses (raw distances let light through walls thinner than the voxel);
// not in tier 0, where dilation made shadow rays leaving a wall at grazing
// angles shadow the wall itself.
const DILATION_RAMP: f32 = 0.1;

fn sun_visibility(
    clipmap: Clipmap,
    tier0: texture_3d<f32>,
    tier1: texture_3d<f32>,
    tier2: texture_3d<f32>,
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
        let tier = tier_at(clipmap, p);
        if tier == OUTSIDE {
            break;
        }
        let raw = sample_tier(clipmap, tier0, tier1, tier2, s, p, tier);
        // Dilation catches thin walls along the ray, not the surface it
        // leaves: ramped in with distance travelled (full tier-1 dilation
        // from 2.5 m), a ray starting near a surface isn't shadowed by it
        // the moment it crosses into a coarse tier. Surfaces lit at under
        // ~6 degrees still self-shadow, where cos is ~0.1 anyway.
        let dilation = min(select(0.0, hit_eps(clipmap, tier), tier > 0u), DILATION_RAMP * t);
        if raw <= dilation {
            return 0.0;
        }
        // The triangulation assumes consecutive empty spheres overlap: the
        // step equals the previous sphere's radius, which keeps y <= raw.
        // So it runs on raw distances and the step actually taken, and the
        // dilation comes off the resulting clearance. (Triangulating the
        // dilated distance against raw-sized steps let y overshoot it once
        // a ray entered a coarse tier, zeroing the clearance: every sun ray
        // that left tier 0 read as fully shadowed.)
        let y = min(raw * raw / (2.0 * previous), raw);
        let d = sqrt(raw * raw - y * y) - dilation;
        res = min(res, k * d / max(t - y, 1e-4));
        let step = max(raw, MIN_STEP);
        previous = step;
        t += step;
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}
