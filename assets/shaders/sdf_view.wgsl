// Debug view: sphere-traces the SDF clipmap from the camera. Writes a
// shaded image and, per pixel, the hit distance along the (unit) ray, or -1
// for a miss, which is what gets compared against Mitsuba's depth.
// Tiers, sampling and hit thresholds come from clipmap_sample.wgsl.

#import s_rc::clipmap_sample::{Clipmap, hit_eps, march, sample_tier, tier_at}

struct View {
    camera: vec3<f32>,
    width: u32,
    // Scaled by tan(fov_y / 2) * aspect, so pixel rays are
    // forward + ndc.x * right + ndc.y * up.
    right: vec3<f32>,
    height: u32,
    // Scaled by tan(fov_y / 2).
    up: vec3<f32>,
    // 0 = normals, 1 = which tier answered, 2 = step count, 3 = direct
    // light, analytic with SDF shadow rays, 4 = light carried by the 3D
    // cascades (both white albedo, so irradiance / pi).
    mode: u32,
    forward: vec3<f32>,
    resolution: u32,
    // Each tier's window origin (world voxel coordinates) in xyz.
    origins: array<vec4<i32>, 3>,
    voxels: vec4<f32>,
    // Cascade 0's probe window (see cascades3d.wgsl).
    c0_window: u32,
    c0_radius: f32,
    c0_tag_base: u32,
    spacing0: f32,
    light_count: u32,
}

struct Light {
    center: vec3<f32>,
    radius: f32,
    radiance: vec3<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var tier0: texture_3d<f32>;
@group(0) @binding(2) var tier1: texture_3d<f32>;
@group(0) @binding(3) var tier2: texture_3d<f32>;
@group(0) @binding(4) var clipmap_sampler: sampler;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<storage, read_write> depth: array<f32>;
// Cascade 0's radiance (16 directions per probe, rgba16float packed), its
// slot tags, and per direction cell 16 sub-directions with solid angles.
@group(0) @binding(7) var<storage, read> cascade0: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read> tags: array<u32>;
@group(0) @binding(9) var<storage, read> shading_subsamples: array<vec4<f32>>;
// Linear radiance per pixel in lighting mode, for exporting.
@group(0) @binding(10) var<storage, read_write> radiance_out: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> lights: array<Light>;

const PI: f32 = 3.14159265359;
const C0_DIRECTIONS: u32 = 16u;
const SUBSAMPLES_PER_DIRECTION: u32 = 16u;

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

// --------------------------------------------------------- direct light

// Reference: fraction of a sphere light visible from `start` from hard
// shadow rays to 16 points spread over the light's disk as seen from there
// (a fixed sunflower pattern, area-uniform), each using the clipmap's
// conservative hit rule. 16x the cost of shadow_aaltonen, for the same
// accuracy against Mitsuba (see SHADOW_METHOD).
const SHADOW_RAYS: u32 = 16u;

fn shadow_visibility(start: vec3<f32>, light: Light) -> f32 {
    let axis = normalize(light.center - start);
    let helper = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    let u = normalize(cross(axis, helper));
    let v = cross(axis, u);
    var visible = 0u;
    for (var i = 0u; i < SHADOW_RAYS; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(SHADOW_RAYS)) * light.radius;
        let angle = f32(i) * 2.39996323;
        let target_point = light.center + r * (cos(angle) * u + sin(angle) * v);
        let toward = target_point - start;
        let len = length(toward);
        let span = len - light.radius;
        if span <= 0.0 || !march(clipmap(), tier0, tier1, tier2, clipmap_sampler, start, toward / len, span).hit {
            visible += 1u;
        }
    }
    return f32(visible) / f32(SHADOW_RAYS);
}

// Aaltonen's estimator (GDC 2018, "Claybook"; as in Quilez's soft shadows
// article): along a shadow ray to the light's center, triangulate from the
// previous and current step distances where the ray passes closest to the
// occluder, instead of only testing at step points, and compare that
// clearance with the light's angular size (k = 1 / tan(angular radius)).
// In coarse tiers, distances are to the surface dilated by the tier's hit
// threshold (the conservative surface every hard hit uses; raw distances let
// light through walls thinner than the voxel).
fn shadow_aaltonen(start: vec3<f32>, light: Light) -> f32 {
    let toward = light.center - start;
    let len = length(toward);
    let dir = toward / len;
    let span = len - light.radius;
    let k = sqrt(max(len * len - light.radius * light.radius, 0.0)) / light.radius;
    var res = 1.0;
    var previous = 1e20;
    var t = 0.0;
    for (var i = 0u; i < 256u && t < span; i++) {
        let p = start + dir * t;
        let tier = tier_at(clipmap(), p);
        if tier == 3u {
            break;
        }
        let raw = sample_at(p, tier);
        // Dilate only in coarse tiers, where thin walls would otherwise
        // vanish; in tier 0 the dilated surface makes a shadow ray leaving
        // a wall at a grazing angle shadow that wall itself.
        let h = raw - select(0.0, hit_eps(clipmap(), tier), tier > 0u);
        if h <= 0.0 {
            return 0.0;
        }
        let y = h * h / (2.0 * previous);
        let d = sqrt(max(h * h - y * y, 0.0));
        res = min(res, k * d / max(t - y, 1e-4));
        previous = h;
        t += max(raw, 0.01);
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// 0 = 16 disk shadow rays (reference), 1 = Aaltonen's estimator.
// Fixed camera inside the room vs Mitsuba (1024 spp):
// - 16 rays: rel. RMSE 13.9%, +3.0%; the only pixels lit where Mitsuba is
//   dark are within 11 px of its lit region (penumbra edges);
// - Aaltonen, dilated in coarse tiers only: 14.1%, -2.4%, none; one march;
// - Aaltonen dilated in tier 0 too: 23.6%, -9.5%, banded self-shadowing;
// - Quilez's earlier symmetric-penumbra variant (no triangulation): leaked
//   faint light into fully shadowed regions.
const SHADOW_METHOD: u32 = 1u;

// Direct irradiance at p (normal n) from the sphere lights: a uniformly
// emitting sphere of angular radius alpha above the horizon gives
// radiance * pi * sin^2(alpha) * cos(theta), times its visible fraction
// (shadow_visibility) along a ray toward its center, started clear of the
// surface (past twice tier 0's hit threshold).
fn direct_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let start = p + n * 0.07;
    var e = vec3(0.0);
    for (var i = 0u; i < view.light_count; i++) {
        let light = lights[i];
        let to_light = light.center - p;
        let dist = length(to_light);
        let cos_theta = dot(n, to_light / dist);
        if dist <= light.radius || cos_theta <= 0.0 {
            continue;
        }
        let sin_alpha = light.radius / dist;
        let visible = select(shadow_visibility(start, light), shadow_aaltonen(start, light), SHADOW_METHOD == 1u);
        e += light.radiance * PI * sin_alpha * sin_alpha * cos_theta * visible;
    }
    return e;
}

// ------------------------------------------------ shading from cascades

fn pack_tag(world: vec3<i32>) -> u32 {
    let w = vec3<u32>(world + 512) & vec3(1023u);
    return (w.x | (w.y << 10u) | (w.z << 20u)) + 1u;
}

fn c0_slot(world: vec3<i32>) -> u32 {
    let size = i32(view.c0_window);
    let s = vec3<u32>((world % size + size) % size);
    return s.x + view.c0_window * (s.y + view.c0_window * s.z);
}

// Cosine-weighted irradiance at one cascade-0 probe for normal n, weighting
// each direction cell by the cosine across its sub-directions.
fn probe_irradiance(slot: u32, n: vec3<f32>) -> vec3<f32> {
    var e = vec3(0.0);
    for (var cell = 0u; cell < C0_DIRECTIONS; cell++) {
        let v = cascade0[slot * C0_DIRECTIONS + cell];
        let radiance = vec4(unpack2x16float(v.x), unpack2x16float(v.y)).rgb;
        var weight = 0.0;
        for (var i = 0u; i < SUBSAMPLES_PER_DIRECTION; i++) {
            let sub = shading_subsamples[cell * SUBSAMPLES_PER_DIRECTION + i];
            weight += max(dot(n, sub.xyz), 0.0) * sub.w;
        }
        e += radiance * weight;
    }
    return e;
}

// Irradiance at surface point p with normal n: trilinear over the 8
// cascade-0 probes around p, sampled half a spacing off the surface. As in
// the 2D testbed, probes that aren't live, sit inside geometry, or are behind
// the surface drop out and the rest are renormalized; with none, black
// (missing coverage loses energy rather than leaking).
fn cascade_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let s = view.spacing0;
    let g = (p + n * 0.5 * s) / s - 0.5;
    let base = vec3<i32>(floor(g));
    let f = g - floor(g);
    var e = vec3(0.0);
    var total = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let c = vec3(k & 1u, (k >> 1u) & 1u, k >> 2u);
        let world = base + vec3<i32>(c);
        let center = (vec3<f32>(world) + 0.5) * s;
        let slot = c0_slot(world);
        let live = distance(center, view.camera) <= view.c0_radius
            && tags[view.c0_tag_base + slot] == pack_tag(world);
        let tier = tier_at(clipmap(), center);
        let empty = tier != 3u && sample_at(center, tier) > 0.0;
        let w3 = mix(1.0 - f, f, vec3<f32>(c));
        let w = w3.x * w3.y * w3.z;
        if live && empty && dot(center - p, n) > 0.0 && w > 0.0 {
            e += w * probe_irradiance(slot, n);
            total += w;
        }
    }
    return select(vec3(0.0), e / total, total > 0.0);
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
            case 2u: { color = vec3(f32(steps) / 64.0, f32(steps) / 256.0, 0.1); }
            default: {
                let p = view.camera + dir * t;
                let n = normal_at(p, tier);
                let radiance = select(cascade_irradiance(p, n), direct_irradiance(p, n), view.mode == 3u) / PI;
                radiance_out[id.y * view.width + id.x] = vec4(radiance, 1.0);
                // Same tonemap as tools/mitsuba_render.py's PNG.
                color = 1.0 - exp(-radiance * 2.0);
            }
        }
    } else if view.mode >= 3u {
        radiance_out[id.y * view.width + id.x] = vec4(0.0);
    }
    textureStore(output, id.xy, vec4(color, 1.0));
    depth[id.y * view.width + id.x] = select(-1.0, t, hit);
}
