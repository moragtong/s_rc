// Debug view: sphere-traces the SDF grid from the camera. Writes a shaded
// image and, per pixel, the hit distance along the (unit) ray, or -1 for a
// miss, which is what gets compared against Mitsuba's depth. Sampling and
// the hit threshold come from clipmap_sample.wgsl.

#import s_rc::clipmap_sample::{Clipmap, grid_uvw, hit_eps, in_grid, march, sample_sdf}

struct View {
    camera: vec3<f32>,
    width: u32,
    // Scaled by tan(fov_y / 2) * aspect, so pixel rays are
    // forward + ndc.x * right + ndc.y * up.
    right: vec3<f32>,
    height: u32,
    // Scaled by tan(fov_y / 2).
    up: vec3<f32>,
    // 0 = normals, 1 = hit distance, 2 = step count, 3 = lighting: direct
    // sun plus indirect from the 3D cascades, times albedo / pi, 4 = the
    // indirect part only; debug: 5 = albedo, 6 = direct sun irradiance,
    // 7 = shadow grid, 8 = color grid (at the hit).
    mode: u32,
    forward: vec3<f32>,
    resolution: u32,
    // The grid window's origin (world voxel coordinates) in xyz.
    origin: vec4<i32>,
    voxel: f32,
    // Cascade 0's probe window (see cascades3d.wgsl).
    c0_window: u32,
    c0_radius: f32,
    c0_tag_base: u32,
    spacing0: f32,
    sun_toward: vec3<f32>,
    sun_k: f32,
    sun_irradiance: vec3<f32>,
    // DEBUG (RC3D_SHADE_DEBUG): 1 = no probe dropout, 2 = nearest probe.
    shade_debug: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var clipmap_sampler: sampler;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> depth: array<f32>;
// Cascade 0's radiance (16 directions per probe, rgba16float packed), its
// slot tags, and per direction cell 16 sub-directions with solid angles.
@group(0) @binding(5) var<storage, read> cascade0: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> tags: array<u32>;
@group(0) @binding(7) var<storage, read> shading_subsamples: array<vec4<f32>>;
// Linear radiance per pixel in lighting mode, for exporting.
@group(0) @binding(8) var<storage, read_write> radiance_out: array<vec4<f32>>;
@group(0) @binding(9) var albedo: texture_3d<f32>;
@group(0) @binding(10) var shadow: texture_3d<f32>;
@group(0) @binding(11) var color: texture_3d<f32>;

const PI: f32 = 3.14159265359;
const C0_DIRECTIONS: u32 = 16u;
const SUBSAMPLES_PER_DIRECTION: u32 = 16u;

fn clipmap() -> Clipmap {
    return Clipmap(view.origin, view.voxel, view.resolution);
}

fn sample_at(p: vec3<f32>) -> f32 {
    return sample_sdf(clipmap(), sdf, clipmap_sampler, p);
}

fn normal_at(p: vec3<f32>) -> vec3<f32> {
    let e = 0.5 * view.voxel;
    return normalize(vec3(
        sample_at(p + vec3(e, 0.0, 0.0)) - sample_at(p - vec3(e, 0.0, 0.0)),
        sample_at(p + vec3(0.0, e, 0.0)) - sample_at(p - vec3(0.0, e, 0.0)),
        sample_at(p + vec3(0.0, 0.0, e)) - sample_at(p - vec3(0.0, 0.0, e)),
    ));
}

fn albedo_at(p: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(albedo, clipmap_sampler, grid_uvw(clipmap(), p), 0.0).rgb;
}

// A march stops anywhere inside the hit threshold, at an offset set by its
// step count, which changes in rings across a surface; shading from there
// shifts its probe weights with it (visible rings). Moving the hit out along
// the normal to the threshold's iso-surface makes it depend only on
// position.
fn on_threshold(hit: vec3<f32>) -> vec3<f32> {
    let eps = hit_eps(clipmap());
    var p = hit;
    for (var i = 0; i < 2; i++) {
        p += normal_at(p) * (eps - sample_at(p));
    }
    return p;
}

// --------------------------------------------------------- direct light

// Direct sun irradiance at p (normal n): irradiance * cos * visibility, the
// visibility read from the shadow grid (no per-pixel shadow rays). Looked
// up half a voxel out along the normal, so the trilinear footprint stays on
// the lit side: voxels inside a wall may hold the other face's visibility.
fn direct_sun(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let cos_theta = dot(n, view.sun_toward);
    if cos_theta <= 0.0 {
        return vec3(0.0);
    }
    return view.sun_irradiance * cos_theta * shadow_at(p, n);
}

fn shadow_at(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let q = p + n * 0.5 * view.voxel;
    if !in_grid(clipmap(), q) {
        return 1.0;
    }
    return textureSampleLevel(shadow, clipmap_sampler, grid_uvw(clipmap(), q), 0.0).r;
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
    // The line test starts 7 cm off the surface, measured in distance-field
    // value (not from p, whose offset inside the hit threshold varies with
    // the march).
    let start = p + n * max(0.07 - sample_at(p), 0.0);
    var e = vec3(0.0);
    var total = 0.0;
    var best = vec3(0.0);
    var best_w = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3(k & 1u, (k >> 1u) & 1u, k >> 2u);
        let world = base + vec3<i32>(corner);
        let center = (vec3<f32>(world) + 0.5) * s;
        let slot = c0_slot(world);
        let live = distance(center, view.camera) <= view.c0_radius
            && tags[view.c0_tag_base + slot] == pack_tag(world);
        let empty = in_grid(clipmap(), center) && sample_at(center) > 0.0;
        let w3 = mix(1.0 - f, f, vec3<f32>(corner));
        let w = w3.x * w3.y * w3.z;
        // In front of the line test's start, and nothing in between.
        let toward = center - start;
        let span = length(toward);
        let clear = dot(toward, n) > 0.0
            && !march(clipmap(), sdf, clipmap_sampler, start, toward / span, span).hit;
        let accepted = select(live && empty && clear, live, view.shade_debug == 1u);
        if accepted && w > 0.0 {
            if view.shade_debug == 2u {
                if w > best_w {
                    best_w = w;
                    best = probe_irradiance(slot, n);
                }
            } else {
                e += w * probe_irradiance(slot, n);
                total += w;
            }
        }
    }
    if view.shade_debug == 2u {
        return best;
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

    let m = march(clipmap(), sdf, clipmap_sampler, view.camera, dir, 1e9);
    let hit = m.hit;
    let t = m.t;

    var shaded = vec3(0.05, 0.05, 0.08);
    if hit {
        let at = view.camera + dir * t;
        switch view.mode {
            case 0u: { shaded = normal_at(at) * 0.5 + 0.5; }
            case 1u: { shaded = vec3(1.0 - exp(-t * 0.1)); }
            case 2u: { shaded = vec3(f32(m.steps) / 64.0, f32(m.steps) / 256.0, 0.1); }
            case 5u: { shaded = albedo_at(at); }
            case 6u: { shaded = 1.0 - exp(-direct_sun(at, normal_at(at)) * 0.5); }
            case 7u: { shaded = vec3(textureSampleLevel(shadow, clipmap_sampler, grid_uvw(clipmap(), at), 0.0).r); }
            case 8u: { shaded = 1.0 - exp(-textureSampleLevel(color, clipmap_sampler, grid_uvw(clipmap(), at), 0.0).rgb * 2.0); }
            default: {
                let p = on_threshold(at);
                let n = normal_at(p);
                var irradiance = cascade_irradiance(p, n);
                // (modes 3 and 4 only reach here via the default arm)
                if view.mode == 3u {
                    irradiance += direct_sun(p, n);
                }
                let radiance = albedo_at(p) / PI * irradiance;
                radiance_out[id.y * view.width + id.x] = vec4(radiance, 1.0);
                // Same tonemap as tools/mitsuba_render.py's PNG.
                shaded = 1.0 - exp(-radiance * 2.0);
            }
        }
    } else if view.mode == 3u || view.mode == 4u {
        radiance_out[id.y * view.width + id.x] = vec4(0.0);
    }
    textureStore(output, id.xy, vec4(shaded, 1.0));
    depth[id.y * view.width + id.x] = select(-1.0, t, hit);
}
