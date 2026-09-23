// Debug view: sphere-traces the SDF clipmap from the camera. Writes a
// shaded image and, per pixel, the hit distance along the (unit) ray, or -1
// for a miss, which is what gets compared against Mitsuba's depth.

struct View {
    camera: vec3<f32>,
    width: u32,
    // Scaled by tan(fov_y / 2) * aspect, so pixel rays are
    // forward + ndc.x * right + ndc.y * up.
    right: vec3<f32>,
    height: u32,
    // Scaled by tan(fov_y / 2).
    up: vec3<f32>,
    // 0 = normals, 1 = which tier answered, 2 = step count.
    mode: u32,
    forward: vec3<f32>,
    resolution: u32,
    // Each tier's window origin (world voxel coordinates) in xyz.
    origins: array<vec4<i32>, 3>,
    voxels: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var tier0: texture_3d<f32>;
@group(0) @binding(2) var tier1: texture_3d<f32>;
@group(0) @binding(3) var tier2: texture_3d<f32>;
@group(0) @binding(4) var clipmap_sampler: sampler;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<storage, read_write> depth: array<f32>;

const MAX_STEPS: u32 = 256u;
const MIN_STEP: f32 = 0.01;

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

fn hit_eps(t: u32) -> f32 {
    return select(TIER0_HIT_FRACTION, 0.5, t > 0u) * view.voxels[t];
}

// Inside tier t's window, with a voxel and a half of margin so trilinear
// filtering never mixes in a texel from the far side of the torus.
fn in_tier(p: vec3<f32>, t: u32) -> bool {
    let voxel = view.voxels[t];
    let origin = vec3<f32>(view.origins[t].xyz);
    let lo = (origin + 1.5) * voxel;
    let hi = (origin + f32(view.resolution) - 1.5) * voxel;
    return all(p > lo) && all(p < hi);
}

// Toroidal: world voxel v's center, (v + 0.5) * voxel, maps to texel
// v mod resolution's center, and repeat addressing does the wrap.
fn sample_tier(p: vec3<f32>, t: u32) -> f32 {
    let uvw = p / (view.voxels[t] * f32(view.resolution));
    switch t {
        case 0u: { return textureSampleLevel(tier0, clipmap_sampler, uvw, 0.0).r; }
        case 1u: { return textureSampleLevel(tier1, clipmap_sampler, uvw, 0.0).r; }
        default: { return textureSampleLevel(tier2, clipmap_sampler, uvw, 0.0).r; }
    }
}

// Finest tier containing p; 3 = outside the clipmap.
fn tier_at(p: vec3<f32>) -> u32 {
    for (var t = 0u; t < 3u; t++) {
        if in_tier(p, t) {
            return t;
        }
    }
    return 3u;
}

fn normal_at(p: vec3<f32>, t: u32) -> vec3<f32> {
    let e = 0.5 * view.voxels[t];
    return normalize(vec3(
        sample_tier(p + vec3(e, 0.0, 0.0), t) - sample_tier(p - vec3(e, 0.0, 0.0), t),
        sample_tier(p + vec3(0.0, e, 0.0), t) - sample_tier(p - vec3(0.0, e, 0.0), t),
        sample_tier(p + vec3(0.0, 0.0, e), t) - sample_tier(p - vec3(0.0, 0.0, e), t),
    ));
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height {
        return;
    }
    let ndc = vec2(
        (f32(id.x) + 0.5) / f32(view.width) * 2.0 - 1.0,
        1.0 - (f32(id.y) + 0.5) / f32(view.height) * 2.0,
    );
    let dir = normalize(view.forward + ndc.x * view.right + ndc.y * view.up);

    var t = 0.0;
    var hit = false;
    var tier = 3u;
    var steps = 0u;
    for (; steps < MAX_STEPS; steps++) {
        let p = view.camera + dir * t;
        tier = tier_at(p);
        if tier == 3u {
            break;
        }
        let d = sample_tier(p, tier);
        if d < hit_eps(tier) {
            hit = true;
            break;
        }
        t += max(d, MIN_STEP);
    }

    var color = vec3(0.05, 0.05, 0.08);
    if hit {
        switch view.mode {
            case 0u: { color = normal_at(view.camera + dir * t, tier) * 0.5 + 0.5; }
            case 1u: { color = array(vec3(0.9, 0.3, 0.2), vec3(0.3, 0.8, 0.3), vec3(0.3, 0.4, 0.9))[tier]; }
            default: { color = vec3(f32(steps) / 64.0, f32(steps) / 256.0, 0.1); }
        }
    }
    textureStore(output, id.xy, vec4(color, 1.0));
    depth[id.y * view.width + id.x] = select(-1.0, t, hit);
}
