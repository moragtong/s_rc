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
// Each dispatch describes its own probes and children in Params.

#import s_rc::clipmap_sample::{Clipmap, grid_uvw, in_grid, march, sample_sdf}

struct Params {
    clipmap: Clipmap,
    camera: vec3<f32>,
    // Cascade index (for the remarch counters).
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
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    // Offset of the children's cells in solid_angles.
    child_angle_base: u32,
    // DEBUG: 1 = no far field.
    debug: u32,
    // Frame counter: dirty flags hold the frame they were set in.
    frame: u32,
    // 1 = every live probe is dirty (e.g. the sun changed).
    force: u32,
    // DEBUG: with debug 3, the one slot debug 2 applies to.
    test_slot: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var clipmap_sampler: sampler;
// The grid's unshadowed surface color and sun visibility.
@group(0) @binding(3) var color: texture_3d<f32>;
@group(0) @binding(4) var shadow: texture_3d<f32>;
// The children's radiance (read) and this pass's (written): rgba16float
// packed in two u32s.
@group(0) @binding(5) var<storage, read> cascade_src: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> cascade_dst: array<vec2<u32>>;
// Cached hit fractions, two unorm16 per word (1.0 = miss).
@group(0) @binding(7) var<storage, read_write> hits: array<u32>;
// Which world probe each slot's cached hits belong to (0 = none).
@group(0) @binding(8) var<storage, read_write> tags: array<u32>;
// Per cascade, probes to (re)trace this frame: their count, then their
// slots in work_list (indexed like tags), and the indirect dispatch
// arguments for tracing them (copied out to the indirect buffer).
@group(0) @binding(9) var<storage, read_write> work_count: array<atomic<u32>>;
// Solid angle of every octahedral cell, all cascades concatenated.
@group(0) @binding(10) var<storage, read> solid_angles: array<f32>;
// Per slot (indexed like tags): bit k set if trilinear child k is reachable,
// i.e. in empty space with a clear straight line from this probe.
@group(0) @binding(11) var<storage, read_write> child_masks: array<u32>;
@group(0) @binding(12) var<storage, read_write> work_list: array<u32>;
@group(0) @binding(13) var<storage, read_write> dispatch_args: array<u32>;
// Cascade 0's direction cells as sub-directions with solid angles (16 per
// cell), and per cascade-0 slot its irradiance as L2 spherical harmonics:
// 9 rgb coefficients in 7 vec4s (27 floats and a pad).
@group(0) @binding(14) var<storage, read> shading_subsamples: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> probe_sh: array<vec4<f32>>;
// Incremental gathering. Per slot (indexed like tags), the frame its probe
// last changed: traced, or one of its children changed. Per cascade, this
// frame's changed probes (count, then slots, indexed like tags), gathered
// by indirect dispatch; the SH projection uses cascade 0's list.
@group(0) @binding(16) var<storage, read_write> dirty: array<u32>;
@group(0) @binding(17) var<storage, read_write> gather_count: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> gather_list: array<u32>;

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
    let m = march(params.clipmap, sdf, clipmap_sampler, seg.a, delta / len, len);
    if !m.hit || m.t >= len {
        return 1.0;
    }
    return min(m.t / len, 65534.0 / 65535.0);
}

// Light leaving the surface hit at p by a ray travelling along dir: color *
// sun visibility. Looked up half a voxel back along the ray, so the
// trilinear footprint stays on the side the ray sees (voxels inside
// geometry carry no color; see clipmap_compose.wgsl).
fn surface_radiance(hit: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p = hit - dir * 0.5 * params.clipmap.voxel;
    if !in_grid(params.clipmap, p) {
        return vec3(0.0);
    }
    let uvw = grid_uvw(params.clipmap, p);
    return textureSampleLevel(color, clipmap_sampler, uvw, 0.0).rgb
        * textureSampleLevel(shadow, clipmap_sampler, uvw, 0.0).r;
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
    cascade_dst[i] = vec2(pack2x16float(r.xy), pack2x16float(r.zw));
}

fn merge(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    return vec4(near.rgb + near.a * far.rgb, near.a * far.a);
}

// Per slot: a live probe whose cached hits belong to another world probe
// (or none) goes on this frame's work list. Tracing then runs over the list
// only, not over every ray of every window just to find them.
@compute @workgroup_size(64, 1, 1)
fn select_probes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let slot = linear_index(gid, groups);
    if slot >= params.window * params.window * params.window {
        return;
    }
    let world = slot_world(slot, params.spacing, params.window);
    if probe_valid(world, params.spacing, params.radius) && tags[params.tag_base + slot] != pack_tag(world) {
        work_list[params.tag_base + atomicAdd(&work_count[params.pass_index], 1u)] = slot;
    }
}

// Indirect dispatch arguments for `threads` threads in workgroups of 64,
// spread over y past 32768 workgroups like `dispatch_linear`.
fn store_args(base: u32, threads: u32) {
    let groups = (threads + 63u) / 64u;
    dispatch_args[base] = min(groups, 32768u);
    dispatch_args[base + 1u] = select((groups + 32767u) / 32768u, 1u, groups <= 32768u);
    dispatch_args[base + 2u] = 1u;
}

// Dispatch argument layout: per cascade, trace at 3n, gather at 3(4 + n);
// the SH projection at 3 * 8.
const GATHER_ARGS: u32 = 12u;
const SH_ARGS: u32 = 24u;

// One thread per cascade: arguments for tracing its work list.
@compute @workgroup_size(1, 1, 1)
fn write_args() {
    let rays = params.octa_side * params.octa_side;
    store_args(params.pass_index * 3u, atomicLoad(&work_count[params.pass_index]) * rays);
}

// Per slot, top down (cascade n after n+1): a live probe changed this frame
// if it was traced, or any trilinear child changed (its merge reads them;
// children of live parents are always live, by the window radii). Changed
// probes go on the gather list.
@compute @workgroup_size(64, 1, 1)
fn mark_dirty(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let slot = linear_index(gid, groups);
    if slot >= params.window * params.window * params.window {
        return;
    }
    let world = slot_world(slot, params.spacing, params.window);
    if !probe_valid(world, params.spacing, params.radius) {
        return;
    }
    var changed = params.force != 0u || dirty[params.tag_base + slot] == params.frame;
    if !changed && has_children() {
        let t = trilinear(probe_center(world, params.spacing), params.child_spacing);
        for (var k = 0u; k < 8u; k++) {
            let child_slot = world_slot(t.base + corner(k), params.child_window);
            if dirty[params.child_tag_base + child_slot] == params.frame {
                changed = true;
            }
        }
    }
    if changed {
        dirty[params.tag_base + slot] = params.frame;
        gather_list[params.tag_base + atomicAdd(&gather_count[params.pass_index], 1u)] = slot;
    }
}

// One thread per cascade, after its marking: arguments for gathering its
// changed probes (and, for cascade 0, for projecting them to SH).
@compute @workgroup_size(1, 1, 1)
fn write_gather_args() {
    let count = atomicLoad(&gather_count[params.pass_index]);
    store_args(GATHER_ARGS + params.pass_index * 3u, count * params.octa_side * params.octa_side);
    if params.pass_index == 0u {
        store_args(SH_ARGS, count);
    }
}

@compute @workgroup_size(64, 1, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = linear_index(gid, groups);
    let rays = params.octa_side * params.octa_side;
    let entry = id / rays;
    if entry >= atomicLoad(&work_count[params.pass_index]) {
        return;
    }
    let ray = id % rays;
    let slot = work_list[params.tag_base + entry];
    let ray_slot = slot * rays + ray;
    let world = slot_world(slot, params.spacing, params.window);
    let count = segment_count();
    for (var k = 0u; k < count; k += 2u) {
        let first = march_fraction(segment(world, ray, k));
        var second = 1.0;
        if k + 1u < count {
            second = march_fraction(segment(world, ray, k + 1u));
        }
        hits[hit_word(ray_slot, k)] = pack2x16unorm(vec2(first, second));
    }
    if ray == 0u {
        dirty[params.tag_base + slot] = params.frame;
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
        if !in_grid(params.clipmap, child) || sample_sdf(params.clipmap, sdf, clipmap_sampler, child) <= 0.0 {
            continue;
        }
        let delta = child - center;
        let len = length(delta);
        if len < 1e-4 || !march(params.clipmap, sdf, clipmap_sampler, center, delta / len, len).hit {
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
// block (2x2 between cascades).
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

// Over this frame's changed probes only (the gather list); the others keep
// last frame's radiance, in this cascade's own buffer.
@compute @workgroup_size(64, 1, 1)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = linear_index(gid, groups);
    let rays = params.octa_side * params.octa_side;
    let entry = index / rays;
    if entry >= atomicLoad(&gather_count[params.pass_index]) {
        return;
    }
    let ray = index % rays;
    let slot = gather_list[params.tag_base + entry];
    let id = vec3(slot * rays + ray, 0u, 0u);
    let world = slot_world(slot, params.spacing, params.window);
    if !has_children() {
        var top = cached_near(id.x, 0u, segment(world, ray, 0u));
        // DEBUG 2 (RC3D_TEST_FRAME): double the top cascade's radiance, to
        // follow a change down the hierarchy.
        if params.debug == 2u || (params.debug == 3u && slot == params.test_slot) {
            top = vec4(top.rgb * 2.0, top.a);
        }
        store_dst(id.x, top);
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

// Real L2 spherical harmonics at unit direction d.
fn sh9(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
        1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y),
    );
}

const SH_WORDS: u32 = 7u;
const SUBSAMPLES_PER_DIRECTION: u32 = 16u;

// After cascade 0's gather (cascade_dst holds it): per changed cascade-0 probe,
// its radiance projected onto L2 spherical harmonics, the cells integrated
// over their sub-directions, and convolved with the clamped cosine (Ramamoorthi
// & Hanrahan: pi, 2pi/3, pi/4 per band), so shading evaluates irradiance
// for any normal from 27 numbers instead of cosine-weighting 16 cells of 16
// sub-directions per probe per pixel.
@compute @workgroup_size(64, 1, 1)
fn project_sh(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let entry = linear_index(gid, groups);
    if entry >= atomicLoad(&gather_count[0]) {
        return;
    }
    let slot = gather_list[params.tag_base + entry];
    let rays = params.octa_side * params.octa_side;
    var c: array<vec3<f32>, 9>;
    for (var cell = 0u; cell < rays; cell++) {
        let v = cascade_dst[slot * rays + cell];
        let radiance = vec4(unpack2x16float(v.x), unpack2x16float(v.y)).rgb;
        for (var i = 0u; i < SUBSAMPLES_PER_DIRECTION; i++) {
            let sub = shading_subsamples[cell * SUBSAMPLES_PER_DIRECTION + i];
            let y = sh9(sub.xyz);
            for (var k = 0u; k < 9u; k++) {
                c[k] += radiance * (y[k] * sub.w);
            }
        }
    }
    let band = array<f32, 9>(3.14159265, 2.0943951, 2.0943951, 2.0943951, 0.7853982, 0.7853982, 0.7853982, 0.7853982, 0.7853982);
    var flat: array<f32, 28>;
    for (var k = 0u; k < 9u; k++) {
        flat[k * 3u] = c[k].x * band[k];
        flat[k * 3u + 1u] = c[k].y * band[k];
        flat[k * 3u + 2u] = c[k].z * band[k];
    }
    for (var w = 0u; w < SH_WORDS; w++) {
        probe_sh[slot * SH_WORDS + w] = vec4(flat[w * 4u], flat[w * 4u + 1u], flat[w * 4u + 2u], flat[w * 4u + 3u]);
    }
}
