// 3D radiance cascades in world space, over the SDF clipmap. The 2D
// testbed's design lifted to 3D (see src/bin/rc2d):
// - Cascade n: probes every spacing0 * 2^n m, directions on an octahedral
//   map of side octa_side0 * 2^n (4x per cascade), interval
//   [interval0 * 2^(n-1), interval0 * 2^n] (cascade 0 from 0).
// - Parent cell (i, j) covers child cells (2i..2i+1, 2j..2j+1); cells are
//   weighted by solid angle (octahedral cells aren't equal-area).
// - Merge: near.rgb + near.a * far.rgb, visibility near.a * far.a.
// - 3D bilinear fix: each ray marches one segment per trilinear child, from
//   its own interval start to where that child's interval starts.
// - Probes live in camera-centred windows, mapped toroidally; tracing caches
//   each segment's hit, re-marching only probes that entered their window,
//   and gathering evaluates lighting at the cached hits every frame.
// Direct light only for now: surfaces emit nothing, lights are spheres
// intersected analytically (they'd vanish in coarse clipmap tiers), and the
// spotlight's cone is ignored.

#import s_rc::clipmap_sample::{Clipmap, OUTSIDE, march, sample_tier, tier_at}

struct Params {
    clipmap: Clipmap,
    camera: vec3<f32>,
    cascade: u32,
    cascade_count: u32,
    light_count: u32,
    spacing0: f32,
    interval0: f32,
    octa_side0: u32,
    // This cascade's and its children's windows.
    window: u32,
    radius: f32,
    tag_base: u32,
    hit_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    // Offset of this cascade's (and its children's) cells in solid_angles.
    angle_base: u32,
    child_angle_base: u32,
}

struct Light {
    center: vec3<f32>,
    radius: f32,
    radiance: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tier0: texture_3d<f32>;
@group(0) @binding(2) var tier1: texture_3d<f32>;
@group(0) @binding(3) var tier2: texture_3d<f32>;
@group(0) @binding(4) var clipmap_sampler: sampler;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
// Cascade n+1's radiance (read) and cascade n's (written): rgba16float
// packed in two u32s.
@group(0) @binding(6) var<storage, read> cascade_src: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> cascade_dst: array<vec2<u32>>;
// Cached hit fractions, two unorm16 per word (1.0 = miss).
@group(0) @binding(8) var<storage, read_write> hits: array<u32>;
// Which world probe each slot's cached hits belong to (0 = none).
@group(0) @binding(9) var<storage, read_write> tags: array<u32>;
@group(0) @binding(10) var<storage, read_write> remarched: array<atomic<u32>>;
// Solid angle of every octahedral cell, all cascades concatenated.
@group(0) @binding(11) var<storage, read> solid_angles: array<f32>;
// Per slot (indexed like tags): bit k set if trilinear child k is reachable,
// i.e. in empty space with a clear straight line from this probe.
@group(0) @binding(12) var<storage, read_write> child_masks: array<u32>;

// ------------------------------------------------------------ geometry

fn spacing(n: u32) -> f32 {
    return params.spacing0 * f32(1u << n);
}

fn octa_side(n: u32) -> u32 {
    return params.octa_side0 << n;
}

fn interval(n: u32) -> vec2<f32> {
    let end = params.interval0 * f32(1u << n);
    return vec2(select(end * 0.5, 0.0, n == 0u), end);
}

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn ray_dir(n: u32, ray: u32) -> vec3<f32> {
    let side = octa_side(n);
    return oct_decode((vec2(f32(ray % side), f32(ray / side)) + 0.5) / f32(side));
}

// Child ray k (0..3) of parent ray `ray` of cascade n, in cascade n+1.
fn child_ray(n: u32, ray: u32, k: u32) -> u32 {
    let side = octa_side(n);
    let cell = vec2(ray % side, ray / side) * 2u + vec2(k & 1u, k >> 1u);
    return cell.y * side * 2u + cell.x;
}

// ------------------------------------------------------- probe windows

fn slot_world(slot: u32, n: u32, window: u32) -> vec3<i32> {
    let size = i32(window);
    let s = vec3<i32>(vec3(slot % window, (slot / window) % window, slot / (window * window)));
    let origin = vec3<i32>(floor(params.camera / spacing(n))) - size / 2;
    return origin + ((s - origin) % size + size) % size;
}

fn world_slot(world: vec3<i32>, window: u32) -> u32 {
    let size = i32(window);
    let s = vec3<u32>((world % size + size) % size);
    return s.x + window * (s.y + window * s.z);
}

fn probe_center(world: vec3<i32>, n: u32) -> vec3<f32> {
    return (vec3<f32>(world) + 0.5) * spacing(n);
}

// Packed world coordinates (10 bits each, offset) + 1; 0 means empty.
fn pack_tag(world: vec3<i32>) -> u32 {
    let w = vec3<u32>(world + 512) & vec3(1023u);
    return (w.x | (w.y << 10u) | (w.z << 20u)) + 1u;
}

fn probe_valid(world: vec3<i32>, n: u32, radius: f32) -> bool {
    return distance(probe_center(world, n), params.camera) <= radius
        && all(abs(world) < vec3(511));
}

// ------------------------------------------------------- segments, hits

fn has_children(n: u32) -> bool {
    return n + 1u < params.cascade_count;
}

fn segment_count(n: u32) -> u32 {
    return select(1u, 8u, has_children(n));
}

fn hit_words(n: u32) -> u32 {
    return (segment_count(n) + 1u) / 2u;
}

// World coordinates of cascade n+1's trilinear child k (0..7) of `pos`,
// and the trilinear weights.
struct Trilinear {
    base: vec3<i32>,
    frac: vec3<f32>,
}

fn trilinear(pos: vec3<f32>, n: u32) -> Trilinear {
    let g = pos / spacing(n) - 0.5;
    return Trilinear(vec3<i32>(floor(g)), g - floor(g));
}

fn corner(k: u32) -> vec3<i32> {
    return vec3<i32>(vec3(k & 1u, (k >> 1u) & 1u, k >> 2u));
}

fn corner_weight(t: Trilinear, k: u32) -> f32 {
    let c = vec3<f32>(corner(k));
    let w = mix(1.0 - t.frac, t.frac, c);
    return w.x * w.y * w.z;
}

struct Segment {
    a: vec3<f32>,
    b: vec3<f32>,
}

// Segment k of `ray` from world probe `world` of cascade n.
fn segment(n: u32, world: vec3<i32>, ray: u32, k: u32) -> Segment {
    let center = probe_center(world, n);
    let range = interval(n);
    let dir = ray_dir(n, ray);
    let a = center + dir * range.x;
    if has_children(n) {
        let child = trilinear(center, n + 1u).base + corner(k);
        return Segment(a, probe_center(child, n + 1u) + dir * range.y);
    }
    return Segment(a, center + dir * range.y);
}

fn hit_sphere(o: vec3<f32>, d: vec3<f32>, light: Light) -> f32 {
    let oc = o - light.center;
    let b = dot(oc, d);
    let c = dot(oc, oc) - light.radius * light.radius;
    if c < 0.0 {
        return 0.0; // starts inside
    }
    let disc = b * b - c;
    if disc < 0.0 || b > 0.0 {
        return 1e9;
    }
    return -b - sqrt(disc);
}

// First hit along the segment (clipmap or light), as a fraction of it.
fn march_fraction(seg: Segment) -> f32 {
    let delta = seg.b - seg.a;
    let len = length(delta);
    let dir = delta / len;
    let m = march(params.clipmap, tier0, tier1, tier2, clipmap_sampler, seg.a, dir, len);
    var t = select(1e9, m.t, m.hit);
    for (var i = 0u; i < params.light_count; i++) {
        t = min(t, hit_sphere(seg.a, dir, lights[i]));
    }
    if t >= len {
        return 1.0;
    }
    return min(t / len, 65534.0 / 65535.0);
}

fn hit_word(ray_slot: u32, n: u32, k: u32) -> u32 {
    return params.hit_base + ray_slot * hit_words(n) + k / 2u;
}

// Radiance and visibility along a cached segment. Lighting is evaluated at
// gather time: a hit on a light sees its radiance, a hit on a surface sees
// nothing yet (direct light only).
fn cached_near(ray_slot: u32, n: u32, k: u32, seg: Segment) -> vec4<f32> {
    let fractions = unpack2x16unorm(hits[hit_word(ray_slot, n, k)]);
    let fraction = select(fractions.x, fractions.y, k % 2u == 1u);
    if fraction >= 1.0 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let pos = mix(seg.a, seg.b, fraction);
    for (var i = 0u; i < params.light_count; i++) {
        // The cached fraction is unorm16 of the segment (up to ~0.2 mm over a
        // 12 m segment) and the sphere hit is f32: allow 1 cm, far more than
        // that error and far less than any surface's distance from a light.
        if distance(pos, lights[i].center) <= lights[i].radius + 0.01 {
            return vec4(lights[i].radiance, 0.0);
        }
    }
    return vec4(0.0);
}

// --------------------------------------------------------------- passes

// Big windows need more than 65535 workgroups of 64, so dispatches spread
// over y too (see `dispatch_linear` on the Rust side).
fn linear_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.y * groups.x * 64u + gid.x;
}

fn load_src(i: u32) -> vec4<f32> {
    let v = cascade_src[i];
    return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn store_dst(i: u32, r: vec4<f32>) {
    cascade_dst[i] = vec2(pack2x16float(r.xy), pack2x16float(r.zw));
}

fn merge(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    return vec4(near.rgb + near.a * far.rgb, near.a * far.a);
}

@compute @workgroup_size(64, 1, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let n = params.cascade;
    let rays = octa_side(n) * octa_side(n);
    if id.x >= params.window * params.window * params.window * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(slot, n, params.window);
    if !probe_valid(world, n, params.radius) || tags[params.tag_base + slot] == pack_tag(world) {
        return;
    }
    let count = segment_count(n);
    for (var k = 0u; k < count; k += 2u) {
        let first = march_fraction(segment(n, world, ray, k));
        var second = 1.0;
        if k + 1u < count {
            second = march_fraction(segment(n, world, ray, k + 1u));
        }
        hits[hit_word(id.x, n, k)] = pack2x16unorm(vec2(first, second));
    }
    if ray == 0u {
        atomicAdd(&remarched[n], 1u);
        if has_children(n) {
            child_masks[params.tag_base + slot] = reachable_children(world, n);
        }
    }
}

// Which trilinear children a probe may take its far field from. With probes
// metres apart, many children sit inside a wall, floor or ceiling, or on the
// far side of one; their connecting segments then hit that surface for
// directions whose real ray would pass freely, darkening everything (and
// they report light from where this probe can't be). Like the shading's
// probe filter (and DDGI's visibility weighting), a child only counts if it
// is in empty space with a clear straight line from this probe. Depends only
// on static geometry, so it's cached with the probe's hits.
fn reachable_children(world: vec3<i32>, n: u32) -> u32 {
    let center = probe_center(world, n);
    let t = trilinear(center, n + 1u);
    var mask = 0u;
    for (var k = 0u; k < 8u; k++) {
        let child = probe_center(t.base + corner(k), n + 1u);
        let tier = tier_at(params.clipmap, child);
        if tier == OUTSIDE || sample_tier(params.clipmap, tier0, tier1, tier2, clipmap_sampler, child, tier) <= 0.0 {
            continue;
        }
        let delta = child - center;
        let len = length(delta);
        if len < 1e-4 || !march(params.clipmap, tier0, tier1, tier2, clipmap_sampler, center, delta / len, len).hit {
            mask |= 1u << k;
        }
    }
    return mask;
}

@compute @workgroup_size(64, 1, 1)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let n = params.cascade;
    let slots = params.window * params.window * params.window;
    if id.x >= slots {
        return;
    }
    let world = slot_world(id.x, n, params.window);
    tags[params.tag_base + id.x] = select(0u, pack_tag(world), probe_valid(world, n, params.radius));
}

// Cascade n+1's radiance for parent ray `ray` of cascade n, at child slot:
// the solid-angle-weighted average of its 4 child cells.
fn child_arc(child_slot: u32, n: u32, ray: u32) -> vec4<f32> {
    let rays1 = octa_side(n + 1u) * octa_side(n + 1u);
    var sum = vec4(0.0);
    var weight = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let c = child_ray(n, ray, k);
        let w = solid_angles[params.child_angle_base + c];
        sum += w * load_src(child_slot * rays1 + c);
        weight += w;
    }
    return sum / weight;
}

@compute @workgroup_size(64, 1, 1)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let n = params.cascade;
    let rays = octa_side(n) * octa_side(n);
    if id.x >= params.window * params.window * params.window * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(slot, n, params.window);
    if !probe_valid(world, n, params.radius) {
        store_dst(id.x, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if !has_children(n) {
        store_dst(id.x, cached_near(id.x, n, 0u, segment(n, world, ray, 0u)));
        return;
    }

    // Children that are unreachable (see reachable_children) or outside
    // their window drop out and the rest are renormalized; with none, the
    // far field is lost energy.
    let t = trilinear(probe_center(world, n), n + 1u);
    let reachable = child_masks[params.tag_base + slot];
    var result = vec4(0.0);
    var total = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let child = t.base + corner(k);
        let child_slot = world_slot(child, params.child_window);
        let ready = (reachable & (1u << k)) != 0u
            && probe_valid(child, n + 1u, params.child_radius)
            && tags[params.child_tag_base + child_slot] == pack_tag(child);
        let w = corner_weight(t, k);
        if ready && w > 0.0 {
            let near = cached_near(id.x, n, k, segment(n, world, ray, k));
            result += w * merge(near, child_arc(child_slot, n, ray));
            total += w;
        }
    }
    if total > 0.0 {
        store_dst(id.x, result / total);
    } else {
        store_dst(id.x, merge(cached_near(id.x, n, 0u, segment(n, world, ray, 0u)), vec4(0.0, 0.0, 0.0, 1.0)));
    }
}
