// Debug view: sphere-traces the SDF clipmap from the camera. Writes a
// shaded image and, per pixel, the hit distance along the (unit) ray, or -1
// for a miss, which is what gets compared against Mitsuba's depth.
// Tiers, sampling and hit thresholds come from clipmap_sample.wgsl.

#import s_rc::clipmap_sample::{Clipmap, march, sample_tier}

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

fn clipmap() -> Clipmap {
    return Clipmap(view.origins, view.voxels, view.resolution);
}

fn sample_at(p: vec3<f32>, t: u32) -> f32 {
    return sample_tier(clipmap(), tier0, tier1, tier2, clipmap_sampler, p, t);
}

fn normal_at(p: vec3<f32>, t: u32) -> vec3<f32> {
    let e = 0.5 * view.voxels[t];
    return normalize(vec3(
        sample_at(p + vec3(e, 0.0, 0.0), t) - sample_at(p - vec3(e, 0.0, 0.0), t),
        sample_at(p + vec3(0.0, e, 0.0), t) - sample_at(p - vec3(0.0, e, 0.0), t),
        sample_at(p + vec3(0.0, 0.0, e), t) - sample_at(p - vec3(0.0, 0.0, e), t),
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

    let m = march(clipmap(), tier0, tier1, tier2, clipmap_sampler, view.camera, dir, 1e9);
    let hit = m.hit;
    let t = m.t;
    let tier = m.tier;
    let steps = m.steps;

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
