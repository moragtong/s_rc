// Debug view: sphere-traces the SDF clipmap from the camera. Writes a
// shaded image and, per pixel, the hit distance along the (unit) ray, or -1
// for a miss, which is what gets compared against Mitsuba's depth.
// Tiers, sampling and hit thresholds come from clipmap_sample.wgsl.

#import s_rc::clipmap_sample::{Clipmap, hit_eps, march, sample_tier, sun_visibility, tier_at}

struct View {
    camera: vec3<f32>,
    width: u32,
    // Scaled by tan(fov_y / 2) * aspect, so pixel rays are
    // forward + ndc.x * right + ndc.y * up.
    right: vec3<f32>,
    height: u32,
    // Scaled by tan(fov_y / 2).
    up: vec3<f32>,
    // 0 = normals, 1 = which tier answered, 2 = step count, 3 = lighting:
    // direct sun computed here plus indirect from the 3D cascades, times
    // albedo / pi, 4 = the indirect part only; debug: 5 = albedo, 6 = direct
    // sun irradiance, 7 = shadow clipmap, 8 = color clipmap (at the hit).
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
    sun_toward: vec3<f32>,
    sun_k: f32,
    sun_irradiance: vec3<f32>,
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
@group(0) @binding(11) var albedo0: texture_3d<f32>;
@group(0) @binding(12) var albedo1: texture_3d<f32>;
@group(0) @binding(13) var albedo2: texture_3d<f32>;
@group(0) @binding(14) var shadow0: texture_3d<f32>;
@group(0) @binding(15) var color0: texture_3d<f32>;

fn debug_shadow(p: vec3<f32>, t: u32) -> f32 {
    return textureSampleLevel(shadow0, clipmap_sampler, p / (view.voxels[0] * f32(view.resolution)), 0.0).r;
}

fn debug_color(p: vec3<f32>, t: u32) -> vec3<f32> {
    return textureSampleLevel(color0, clipmap_sampler, p / (view.voxels[0] * f32(view.resolution)), 0.0).rgb;
}

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

// Direct sun irradiance at p (normal n): irradiance * cos * visibility, the
// visibility from the shared Aaltonen estimator, marched from just off the
// surface (past twice tier 0's hit threshold). Computed per pixel here, not
// read from the shadow clipmap: this is where sharp (e.g. dynamic) shadow
// boundaries have to survive.
fn direct_sun(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let cos_theta = dot(n, view.sun_toward);
    if cos_theta <= 0.0 {
        return vec3(0.0);
    }
    let visible = sun_visibility(clipmap(), tier0, tier1, tier2, clipmap_sampler, p + n * 0.07, view.sun_toward, view.sun_k);
    return view.sun_irradiance * cos_theta * visible;
}

fn albedo_at(p: vec3<f32>, t: u32) -> vec3<f32> {
    let uvw = p / (view.voxels[t] * f32(view.resolution));
    switch t {
        case 0u: { return textureSampleLevel(albedo0, clipmap_sampler, uvw, 0.0).rgb; }
        case 1u: { return textureSampleLevel(albedo1, clipmap_sampler, uvw, 0.0).rgb; }
        default: { return textureSampleLevel(albedo2, clipmap_sampler, uvw, 0.0).rgb; }
    }
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
// the 2D testbed, probes that aren't live, sit inside geometry, are behind
// the surface, or have no clear line from it drop out and the rest are
// renormalized; with none, black (missing coverage loses energy rather than
// leaking). The line test is what keeps probes on the far side of a thin
// slab out: e.g. the grid row above a 20 cm ceiling is in empty space and
// in front of a wall point just under the ceiling, but out on the roof.
fn cascade_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let s = view.spacing0;
    let g = (p + n * 0.5 * s) / s - 0.5;
    let base = vec3<i32>(floor(g));
    let f = g - floor(g);
    // The line test starts past the hit threshold of p's tier: from 7 cm,
    // a coarse tier (25 cm threshold) hit the surface itself and dropped
    // every probe.
    let start = p + n * max(0.07, 1.25 * hit_eps(clipmap(), tier_at(clipmap(), p)));
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
        let toward = center - start;
        let span = length(toward);
        // A probe no further out than the start can't have an occluder
        // between it and the surface.
        let clear = dot(toward, n) <= 0.0
            || !march(clipmap(), tier0, tier1, tier2, clipmap_sampler, start, toward / span, span).hit;
        if live && empty && dot(center - p, n) > 0.0 && w > 0.0 && clear {
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
            case 5u: { color = albedo_at(view.camera + dir * t, tier); }
            case 6u: {
                let p = view.camera + dir * t;
                color = 1.0 - exp(-direct_sun(p, normal_at(p, tier)) * 0.5);
            }
            case 7u: { color = vec3(debug_shadow(view.camera + dir * t, tier)); }
            case 8u: { color = 1.0 - exp(-debug_color(view.camera + dir * t, tier) * 2.0); }
            default: {
                let p = view.camera + dir * t;
                let n = normal_at(p, tier);
                var irradiance = cascade_irradiance(p, n);
                // (modes 3 and 4 only reach here via the default arm)
                if view.mode == 3u {
                    irradiance += direct_sun(p, n);
                }
                let radiance = albedo_at(p, tier) / PI * irradiance;
                radiance_out[id.y * view.width + id.x] = vec4(radiance, 1.0);
                // Same tonemap as tools/mitsuba_render.py's PNG.
                color = 1.0 - exp(-radiance * 2.0);
            }
        }
    } else if view.mode == 3u || view.mode == 4u {
        radiance_out[id.y * view.width + id.x] = vec4(0.0);
    }
    textureStore(output, id.xy, vec4(color, 1.0));
    depth[id.y * view.width + id.x] = select(-1.0, t, hit);
}
