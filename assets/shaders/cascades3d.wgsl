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
// The cascades carry indirect light: a ray that hits a surface sees the
// light leaving it, which is the clipmap's unshadowed color times its sun
// visibility (the shadow clipmap), sampled at the hit. Direct sunlight at
// the shading point itself is computed there, not by the cascades.
// Shading levels (layout.rs) run through the same passes: cascade 0 scaled
// up, merging with a cascade whose directions are more than 4x finer.
// Each dispatch describes its own probes and children in Params.

#import s_rc::clipmap_sample::{Clipmap, OUTSIDE, march, sample_tier, tier_at}

struct Params {
    clipmap: Clipmap,
    camera: vec3<f32>,
    // Cascades 0.., then shading levels 1.. (for the remarch counters).
    pass_index: u32,
    // Ray interval, m.
    interval: vec2<f32>,
    spacing: f32,
    octa_side: u32,
    has_children: u32,
    child_spacing: f32,
    child_octa_side: u32,
    // This pass's and its children's windows.
    window: u32,
    radius: f32,
    tag_base: u32,
    hit_base: u32,
    // Where this pass's rays go in cascade_dst.
    dst_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    // Offset of the children's cells in solid_angles.
    child_angle_base: u32,
    // DEBUG: 1 = no far field.
    debug: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tier0: texture_3d<f32>;
@group(0) @binding(2) var tier1: texture_3d<f32>;
@group(0) @binding(3) var tier2: texture_3d<f32>;
@group(0) @binding(4) var clipmap_sampler: sampler;
// The clipmap's unshadowed surface color and sun visibility, per tier.
@group(0) @binding(5) var color0: texture_3d<f32>;
@group(0) @binding(6) var color1: texture_3d<f32>;
@group(0) @binding(7) var color2: texture_3d<f32>;
@group(0) @binding(8) var shadow0: texture_3d<f32>;
@group(0) @binding(9) var shadow1: texture_3d<f32>;
@group(0) @binding(10) var shadow2: texture_3d<f32>;
// The children's radiance (read) and this pass's (written): rgba16float
// packed in two u32s.
@group(0) @binding(11) var<storage, read> cascade_src: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> cascade_dst: array<vec2<u32>>;
// Cached hit fractions, two unorm16 per word (1.0 = miss).
@group(0) @binding(13) var<storage, read_write> hits: array<u32>;
// Which world probe each slot's cached hits belong to (0 = none).
@group(0) @binding(14) var<storage, read_write> tags: array<u32>;
@group(0) @binding(15) var<storage, read_write> remarched: array<atomic<u32>>;
// Solid angle of every octahedral cell, all cascades concatenated.
@group(0) @binding(16) var<storage, read> solid_angles: array<f32>;
// Per slot (indexed like tags): bit k set if trilinear child k is reachable,
// i.e. in empty space with a clear straight line from this probe.
@group(0) @binding(17) var<storage, read_write> child_masks: array<u32>;

// ------------------------------------------------------------ geometry


fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn ray_dir(ray: u32) -> vec3<f32> {
    let side = params.octa_side;
    return oct_decode((vec2(f32(ray % side), f32(ray / side)) + 0.5) / f32(side));
}

// ------------------------------------------------------- probe windows

fn slot_world(slot: u32, spacing: f32, window: u32) -> vec3<i32> {
    let size = i32(window);
    let s = vec3<i32>(vec3(slot % window, (slot / window) % window, slot / (window * window)));
    let origin = vec3<i32>(floor(params.camera / spacing)) - size / 2;
    return origin + ((s - origin) % size + size) % size;
}

fn world_slot(world: vec3<i32>, window: u32) -> u32 {
    let size = i32(window);
    let s = vec3<u32>((world % size + size) % size);
    return s.x + window * (s.y + window * s.z);
}

fn probe_center(world: vec3<i32>, spacing: f32) -> vec3<f32> {
    return (vec3<f32>(world) + 0.5) * spacing;
}

// Packed world coordinates (10 bits each, offset) + 1; 0 means empty.
fn pack_tag(world: vec3<i32>) -> u32 {
    let w = vec3<u32>(world + 512) & vec3(1023u);
    return (w.x | (w.y << 10u) | (w.z << 20u)) + 1u;
}

fn probe_valid(world: vec3<i32>, spacing: f32, radius: f32) -> bool {
    return distance(probe_center(world, spacing), params.camera) <= radius
        && all(abs(world) < vec3(511));
}

// ------------------------------------------------------- segments, hits

fn has_children() -> bool {
    return params.has_children != 0u;
}

fn segment_count() -> u32 {
    return select(1u, 8u, has_children());
}

fn hit_words() -> u32 {
    return (segment_count() + 1u) / 2u;
}

// World coordinates of the child grid's trilinear child k (0..7) of `pos`,
// and the trilinear weights.
struct Trilinear {
    base: vec3<i32>,
    frac: vec3<f32>,
}

fn trilinear(pos: vec3<f32>, spacing: f32) -> Trilinear {
    let g = pos / spacing - 0.5;
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

// Segment k of `ray` from world probe `world`.
fn segment(world: vec3<i32>, ray: u32, k: u32) -> Segment {
    let center = probe_center(world, params.spacing);
    let range = params.interval;
    let dir = ray_dir(ray);
    let a = center + dir * range.x;
    if has_children() {
        let child = trilinear(center, params.child_spacing).base + corner(k);
        return Segment(a, probe_center(child, params.child_spacing) + dir * range.y);
    }
    return Segment(a, center + dir * range.y);
}

// First hit along the segment, as a fraction of it.
fn march_fraction(seg: Segment) -> f32 {
    let delta = seg.b - seg.a;
    let len = length(delta);
    let m = march(params.clipmap, tier0, tier1, tier2, clipmap_sampler, seg.a, delta / len, len);
    if !m.hit || m.t >= len {
        return 1.0;
    }
    return min(m.t / len, 65534.0 / 65535.0);
}

// Light leaving the surface hit at p by a ray travelling along dir: color *
// sun visibility, from the finest tier holding p. Looked up half a voxel
// back along the ray, so the trilinear footprint stays on the side the ray
// sees (voxels inside geometry carry no color; see clipmap_compose.wgsl).
fn surface_radiance(hit: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let hit_tier = tier_at(params.clipmap, hit);
    let p = hit - dir * 0.5 * params.clipmap.voxels[min(hit_tier, 2u)];
    let tier = tier_at(params.clipmap, p);
    let uvw = p / (params.clipmap.voxels[min(tier, 2u)] * f32(params.clipmap.resolutions[min(tier, 2u)]));
    switch tier {
        case 0u: {
            return textureSampleLevel(color0, clipmap_sampler, uvw, 0.0).rgb
                * textureSampleLevel(shadow0, clipmap_sampler, uvw, 0.0).r;
        }
        case 1u: {
            return textureSampleLevel(color1, clipmap_sampler, uvw, 0.0).rgb
                * textureSampleLevel(shadow1, clipmap_sampler, uvw, 0.0).r;
        }
        case 2u: {
            return textureSampleLevel(color2, clipmap_sampler, uvw, 0.0).rgb
                * textureSampleLevel(shadow2, clipmap_sampler, uvw, 0.0).r;
        }
        default: { return vec3(0.0); }
    }
}

fn hit_word(ray_slot: u32, k: u32) -> u32 {
    return params.hit_base + ray_slot * hit_words() + k / 2u;
}

// Radiance and visibility along a cached segment. Lighting is evaluated at
// gather time from the clipmap's color and shadow at the hit.
fn cached_near(ray_slot: u32, k: u32, seg: Segment) -> vec4<f32> {
    let fractions = unpack2x16unorm(hits[hit_word(ray_slot, k)]);
    let fraction = select(fractions.x, fractions.y, k % 2u == 1u);
    if fraction >= 1.0 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    // Hit at its start: the segment begins inside (or on) geometry, e.g. a
    // cascade's interval starting inside a wall. It sees nothing. Looking
    // up color there read the wall's far face through the trilinear
    // footprint: at 6.25 cm voxels, half the indirect light was this.
    if fraction <= 0.0 {
        return vec4(0.0);
    }
    return vec4(surface_radiance(mix(seg.a, seg.b, fraction), normalize(seg.b - seg.a)), 0.0);
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
    cascade_dst[params.dst_base + i] = vec2(pack2x16float(r.xy), pack2x16float(r.zw));
}

fn merge(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    return vec4(near.rgb + near.a * far.rgb, near.a * far.a);
}

@compute @workgroup_size(64, 1, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let rays = params.octa_side * params.octa_side;
    if id.x >= params.window * params.window * params.window * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(slot, params.spacing, params.window);
    if !probe_valid(world, params.spacing, params.radius) || tags[params.tag_base + slot] == pack_tag(world) {
        return;
    }
    let count = segment_count();
    for (var k = 0u; k < count; k += 2u) {
        let first = march_fraction(segment(world, ray, k));
        var second = 1.0;
        if k + 1u < count {
            second = march_fraction(segment(world, ray, k + 1u));
        }
        hits[hit_word(id.x, k)] = pack2x16unorm(vec2(first, second));
    }
    if ray == 0u {
        atomicAdd(&remarched[params.pass_index], 1u);
        if has_children() {
            child_masks[params.tag_base + slot] = reachable_children(world);
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
fn reachable_children(world: vec3<i32>) -> u32 {
    let center = probe_center(world, params.spacing);
    let t = trilinear(center, params.child_spacing);
    var mask = 0u;
    for (var k = 0u; k < 8u; k++) {
        let child = probe_center(t.base + corner(k), params.child_spacing);
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
    let slots = params.window * params.window * params.window;
    if id.x >= slots {
        return;
    }
    let world = slot_world(id.x, params.spacing, params.window);
    tags[params.tag_base + id.x] = select(0u, pack_tag(world), probe_valid(world, params.spacing, params.radius));
}

// The children's radiance for parent ray `ray`, at child slot: the
// solid-angle-weighted average of the child cells it covers, a ratio x ratio
// block (2x2 between cascades; more for a shading level).
fn child_arc(child_slot: u32, ray: u32) -> vec4<f32> {
    let side = params.octa_side;
    let child_side = params.child_octa_side;
    let ratio = child_side / side;
    let first = vec2(ray % side, ray / side) * ratio;
    var sum = vec4(0.0);
    var weight = 0.0;
    for (var j = 0u; j < ratio; j++) {
        for (var i = 0u; i < ratio; i++) {
            let c = (first.y + j) * child_side + first.x + i;
            let w = solid_angles[params.child_angle_base + c];
            sum += w * load_src(child_slot * child_side * child_side + c);
            weight += w;
        }
    }
    return sum / weight;
}

// For a shading level: its children's radiance averaged down to its own
// directions, once per child probe (child_window^3 x rays), so the gather
// reads one value per child instead of ratio^2. The gather then runs with
// child_octa_side = octa_side on this output.
@compute @workgroup_size(64, 1, 1)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let rays = params.octa_side * params.octa_side;
    if id.x >= params.child_window * params.child_window * params.child_window * rays {
        return;
    }
    store_dst(id.x, child_arc(id.x / rays, id.x % rays));
}

@compute @workgroup_size(64, 1, 1)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let rays = params.octa_side * params.octa_side;
    if id.x >= params.window * params.window * params.window * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(slot, params.spacing, params.window);
    if !probe_valid(world, params.spacing, params.radius) {
        store_dst(id.x, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if !has_children() {
        store_dst(id.x, cached_near(id.x, 0u, segment(world, ray, 0u)));
        return;
    }

    // Children that are unreachable (see reachable_children) or outside
    // their window drop out and the rest are renormalized; with none, the
    // far field is lost energy.
    let t = trilinear(probe_center(world, params.spacing), params.child_spacing);
    let reachable = child_masks[params.tag_base + slot];
    var result = vec4(0.0);
    var total = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let child = t.base + corner(k);
        let child_slot = world_slot(child, params.child_window);
        let ready = (reachable & (1u << k)) != 0u
            && probe_valid(child, params.child_spacing, params.child_radius)
            && tags[params.child_tag_base + child_slot] == pack_tag(child);
        let w = corner_weight(t, k);
        if ready && w > 0.0 {
            let near = cached_near(id.x, k, segment(world, ray, k));
            let far = select(child_arc(child_slot, ray), vec4(0.0, 0.0, 0.0, 1.0), params.debug == 1u);
            result += w * merge(near, far);
            total += w;
        }
    }
    if total > 0.0 {
        store_dst(id.x, result / total);
    } else {
        store_dst(id.x, merge(cached_near(id.x, 0u, segment(world, ray, 0u)), vec4(0.0, 0.0, 0.0, 1.0)));
    }
}
