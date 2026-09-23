// 2D radiance cascades testbed, plus a brute-force reference over the
// exact same scene and trace function, so any difference between the two
// is the cascade approximation and nothing else.
//
// Conventions (checked against m4xc.dev "Fundamentals of Radiance Cascades"
// and Osborne & Sannikov 2024, arXiv:2408.14425):
// - Branching factor 4: cascade n has 4^(n+1) rays per probe, probe spacing
//   s0*2^n px (s0 = 1 or 2 px), interval [b*4^n, b*4^(n+1)] with cascade 0
//   starting at 0. Memory per cascade is constant:
//   (W/(s0 2^n))(H/(s0 2^n)) * 4^(n+1) = 4WH/s0^2.
// - Ray i of cascade n covers the same arc as rays 4i..4i+3 of cascade n+1.
// - Merge: near.rgb + near.a * far.rgb, visibility near.a * far.a
//   (a = 1 means the segment hit nothing).
// - Cascade n+1 probes are bilinearly interpolated at cascade n probe
//   centers, with cell-centered coordinates (pos / spacing - 0.5).
// - Tracing and gathering are separate: cascade_trace caches each segment's
//   hit location, re-marching only probes that entered the circle of live
//   probes (see "probe window"), and cascade_gather evaluates lighting at
//   the cached hits every frame.

// Primitives and materials live in rc2d_sdf.wgsl, the room in rc2d_room.wgsl.
#import rc2d::sdf::{SceneSample, sd_box, sd_circle, closer, spot_factor}
#import rc2d::room::room_scene

struct Params {
    size: vec2<u32>,
    cascade: u32,
    cascade_count: u32,
    base_interval: f32,
    // 0 = plain bilinear merge, 1 = bilinear fix.
    merge_mode: u32,
    // 0 = cascades, 1 = reference, 2 = |cascades - reference|.
    view_mode: u32,
    // Frames accumulated into the reference since the last reset.
    ref_frame: u32,
    ref_rays: u32,
    exposure: f32,
    diff_gain: f32,
    // Warm light center; draggable.
    light_pos: vec2<f32>,
    // log2 of the cascade-0 probe spacing s0 in px.
    probe0_log2: u32,
    // Angle added to every ray of every cascade (radians). The same for all
    // cascades, so child arcs still subdivide parent arcs exactly.
    ray_rotation: f32,
    // Diagnostic: 1 = average the resolved cascades over frames (each with
    // a different ray_rotation) instead of showing only this frame.
    rc_accumulate: u32,
    // Frames in rc_history since the last reset; 0 = overwrite.
    rc_frame: u32,
    // Cascades n < this merge each of the 4 child directions with its own
    // near segment and average after; the rest trace one near segment along
    // the parent's center direction and merge it with the pre-averaged
    // children.
    per_direction_levels: u32,
    // 0 = lights and occluders, 1 = room with thin diffuse walls, a
    // spotlight outside its entrance and more lights around the outside.
    scene_id: u32,
    // Diffuse bounces. The reference path-traces this many; the cascades
    // run bounces + 1 passes per frame, see bounce_iteration.
    bounces: u32,
    // Gather pass within this frame. Pass 0 is direct light only; pass
    // k > 0 lights diffuse surfaces from pass k-1's cascade 0 (bounce_src).
    bounce_iteration: u32,
    // Circle of live probes (see "probe window"), and cascade 0's radius,
    // which is also the displayed and measured region.
    circle_center: vec2<f32>,
    // This cascade's, its children's and cascade 0's probe windows.
    window: u32,
    radius: f32,
    tag_base: u32,
    child_window: u32,
    child_radius: f32,
    child_tag_base: u32,
    c0_window: u32,
    c0_radius: f32,
    c0_tag_base: u32,
    // This cascade's first word in `hits`.
    hit_base: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Cascade entries are rgba16float packed into two u32s (pack2x16float), so
// half the bandwidth of vec4<f32> without needing the f16 shader feature.
@group(0) @binding(1) var<storage, read> cascade_src: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> cascade_dst: array<vec2<u32>>;
// rgb = summed radiance, a = number of rays summed.
@group(0) @binding(3) var<storage, read_write> reference: array<vec4<f32>>;
// Two vec4s per pixel, read back on the CPU:
// [2i]   (squared rgb error, squared reference, signed error summed over
//         rgb, reference ray count; negated for pixels inside shapes)
// [2i+1] (reference summed over rgb, cascades summed over rgb,
//         1 if the pixel is on a diffuse wall, 0)
@group(0) @binding(4) var<storage, read_write> error: array<vec4<f32>>;
@group(0) @binding(5) var display: texture_storage_2d<rgba8unorm, write>;
// rgb = summed resolved cascades, a = frame count (rc_accumulate only).
@group(0) @binding(6) var<storage, read_write> rc_history: array<vec4<f32>>;
// Cascade 0 of the previous bounce pass, same packed layout as the cascades.
@group(0) @binding(7) var<storage, read> bounce_src: array<vec2<u32>>;
// Cached hit fractions, two unorm16 per word; see "segments and hits".
@group(0) @binding(8) var<storage, read_write> hits: array<u32>;
// Which world probe each slot's cached hits belong to; see "probe window".
@group(0) @binding(9) var<storage, read_write> tags: array<u32>;
// Probes re-marched this frame, per cascade (cleared by the CPU each frame).
@group(0) @binding(10) var<storage, read_write> remarched: array<atomic<u32>>;

const TAU: f32 = 6.28318530718;
const PI: f32 = 3.14159265359;
const HIT_EPS: f32 = 0.05;
const MAX_STEPS: u32 = 256u;

// ---------------------------------------------------------------- scene

// Pixel units, y down. Emitters also occlude; zero emission = black occluder.
fn scene(p: vec2<f32>) -> SceneSample {
    if params.scene_id == 1u {
        return room_scene(p, params.light_pos);
    }
    var s = SceneSample(1e9, vec3(0.0), 0.0, vec3(0.0));
    // Lights.
    s = closer(s, sd_circle(p, params.light_pos, 22.0), vec3(8.0, 4.5, 1.5));
    s = closer(s, sd_circle(p, vec2(420.0, 100.0), 14.0), vec3(1.5, 3.5, 10.0));
    s = closer(s, sd_box(p, vec2(400.0, 490.0), vec2(80.0, 3.0)), vec3(0.4, 3.0, 0.8));
    // Thin wall right of the warm light's start position: the leak test.
    // Anything warm to its right is light that got interpolated through it.
    s = closer(s, sd_box(p, vec2(200.0, 400.0), vec2(1.5, 75.0)), vec3(0.0));
    // Occluders for penumbrae at varying distances from the lights.
    s = closer(s, sd_box(p, vec2(256.0, 256.0), vec2(60.0, 12.0)), vec3(0.0));
    s = closer(s, sd_circle(p, vec2(170.0, 170.0), 18.0), vec3(0.0));
    s = closer(s, sd_circle(p, vec2(370.0, 200.0), 8.0), vec3(0.0));
    return s;
}

fn scene_normal(p: vec2<f32>) -> vec2<f32> {
    let h = vec2(0.5, 0.0);
    return normalize(vec2(
        scene(p + h.xy).dist - scene(p - h.xy).dist,
        scene(p + h.yx).dist - scene(p - h.yx).dist,
    ));
}

// For a point inside a diffuse wall: the nearest point on its surface and
// the outward normal there. A 2px wall's two pixel columns each get their
// own face.
struct Face {
    pos: vec2<f32>,
    normal: vec2<f32>,
}

fn nearest_face(p: vec2<f32>, dist: f32) -> Face {
    let n = scene_normal(p);
    return Face(p - n * dist, n);
}

struct Hit {
    hit: bool,
    pos: vec2<f32>,
    // Already restricted by the spotlight cone for this ray's direction.
    emission: vec3<f32>,
    albedo: vec3<f32>,
}

// First surface along [t0, t1]. Rays that leave the canvas never re-enter
// it (it's convex), so they miss.
fn trace_hit(origin: vec2<f32>, dir: vec2<f32>, t0: f32, t1: f32) -> Hit {
    let bounds = vec2<f32>(params.size);
    var t = t0;
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = origin + dir * t;
        if any(p < vec2(0.0)) || any(p > bounds) {
            break;
        }
        let s = scene(p);
        if s.dist < HIT_EPS {
            let emission = s.emission * select(1.0, spot_factor(dir), s.spot > 0.0);
            return Hit(true, p, emission, s.albedo);
        }
        t += s.dist;
        if t >= t1 {
            break;
        }
    }
    return Hit(false, vec2(0.0), vec3(0.0), vec3(0.0));
}

// ---------------------------------------------------------- probe window
//
// Probes only exist inside a circle (eventually the camera's region). Each
// cascade stores its probes in a `window` x `window` grid of slots, mapped
// toroidally: world probe w lives in slot w mod window. When the circle
// moves, most probes keep their slot, and so their cached hits; only slots
// whose world probe changed get re-marched. Cascade n's circle is larger
// than n-1's by 1.5 of its own spacings, so every valid probe's 4 bilinear
// children exist.
//
// A slot's tag records which world probe its cached hits belong to
// (packed coordinates + 1; 0 = nothing cached). The trace pass re-marches
// valid slots whose tag doesn't match, and a separate commit pass updates
// the tags afterwards, so no thread reads a tag another thread is writing.

struct Window {
    size: u32,
    radius: f32,
    tag_base: u32,
}

fn self_window() -> Window {
    return Window(params.window, params.radius, params.tag_base);
}

fn child_window() -> Window {
    return Window(params.child_window, params.child_radius, params.child_tag_base);
}

fn c0_window() -> Window {
    return Window(params.c0_window, params.c0_radius, params.c0_tag_base);
}

fn slot_world(slot: vec2<u32>, n: u32, window: u32) -> vec2<i32> {
    // A window spanning the whole canvas grid holds every probe as-is;
    // centering it on the circle would wrap canvas probes out of range.
    if window >= probe_grid(n).x {
        return vec2<i32>(slot);
    }
    let size = i32(window);
    let origin = vec2<i32>(floor(params.circle_center / probe_spacing(n))) - size / 2;
    return origin + ((vec2<i32>(slot) - origin) % size + size) % size;
}

fn world_slot(world: vec2<i32>, window: u32) -> vec2<u32> {
    let size = i32(window);
    return vec2<u32>((world % size + size) % size);
}

fn slot_index(slot: vec2<u32>, window: u32) -> u32 {
    return slot.y * window + slot.x;
}

fn pack_tag(world: vec2<i32>) -> u32 {
    return (u32(world.x) | (u32(world.y) << 16u)) + 1u;
}

// In the canvas probe grid and within this cascade's circle.
fn probe_valid(world: vec2<i32>, n: u32, radius: f32) -> bool {
    if any(world < vec2(0)) || any(world >= vec2<i32>(probe_grid(n))) {
        return false;
    }
    let center = (vec2<f32>(world) + 0.5) * probe_spacing(n);
    return distance(center, params.circle_center) <= radius;
}

// Valid and holding up-to-date data (its tag was committed this frame).
fn probe_ready(world: vec2<i32>, n: u32, window: Window) -> bool {
    return probe_valid(world, n, window.radius)
        && tags[window.tag_base + slot_index(world_slot(world, window.size), window.size)] == pack_tag(world);
}

// -------------------------------------------------------------- bounces

// 2D Lambertian: L_out = (albedo / 2) * integral of L_in cos(theta) over
// the half-circle around the normal. With uniform L_in that's albedo * L_in,
// which is also what the reference's cosine-sampled estimator gives.
// This returns L_out / albedo from cascade-0 probes: the previous pass's
// (bounce_src) while gathering, or this pass's final result (cascade_src, in
// the resolve) to draw wall faces.
//
// Only probes in empty space on the normal's side of the surface count. A
// 2px wall is thinner than 2px probe spacing, so the plain bilinear
// footprint straddles it and would light the inner face with the outer
// face's light. With no valid probe (including outside the circle) the
// bounce is dropped: missing coverage loses energy rather than leaking it.
fn bounce_radiance(pos: vec2<f32>, final_pass: bool) -> vec3<f32> {
    let n = scene_normal(pos);
    let spacing = probe_spacing(0u);
    let window = c0_window();
    // Sample the probe grid half a spacing off the surface. Exactly on it,
    // the footprint can hold only the probe rows centered on the wall's two
    // faces, which the side test rejects, and the face goes black; half a
    // spacing out, one bracketing row is always strictly in front.
    let probes = bilinear(pos + n * 0.5 * spacing, 0u);
    var sum = vec3(0.0);
    var weight = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let world = probes.base + BILINEAR_OFFSETS[k];
        let center = (vec2<f32>(world) + 0.5) * spacing;
        if probe_ready(world, 0u, window) && scene(center).dist > 0.0 && dot(center - pos, n) > 0.0 {
            let slot = slot_index(world_slot(world, window.size), window.size);
            sum += probes.weights[k] * cosine_weighted(slot, n, final_pass);
            weight += probes.weights[k];
        }
    }
    return select(vec3(0.0), sum / weight, weight > 0.0);
}

// (1/2) * integral of L_in cos over the half-circle around n, from one
// cascade-0 probe's 4 arc-averaged rays. Each arc's cosine weight is exact:
// sin(hi) - sin(lo) over the part of the arc inside [-90, 90] degrees.
fn cosine_weighted(slot: u32, n: vec2<f32>, final_pass: bool) -> vec3<f32> {
    let normal_angle = atan2(n.y, n.x);
    var sum = vec3(0.0);
    for (var i = 0u; i < 4u; i++) {
        let d = (f32(i) + 0.5) * TAU / 4.0 + params.ray_rotation - normal_angle;
        let offset = d - TAU * floor((d + PI) / TAU);
        let lo = max(offset - TAU / 8.0, -PI / 2.0);
        let hi = min(offset + TAU / 8.0, PI / 2.0);
        let v = select(bounce_src[slot * 4u + i], cascade_src[slot * 4u + i], final_pass);
        let radiance = vec4(unpack2x16float(v.x), unpack2x16float(v.y)).rgb;
        sum += select(0.0, sin(hi) - sin(lo), hi > lo) * radiance;
    }
    return 0.5 * sum;
}

// ------------------------------------------------------------- cascades

fn ray_count(n: u32) -> u32 {
    return 4u << (2u * n);
}

fn probe_spacing(n: u32) -> f32 {
    return f32(1u << (n + params.probe0_log2));
}

fn probe_grid(n: u32) -> vec2<u32> {
    return params.size >> vec2(n + params.probe0_log2);
}

// Cell-centered bilinear footprint of cascade n's probes at `pos`, in world
// probe coordinates (unclamped).
struct Bilinear {
    base: vec2<i32>,
    weights: vec4<f32>,
}

const BILINEAR_OFFSETS = array(vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1));

fn bilinear(pos: vec2<f32>, n: u32) -> Bilinear {
    let g = pos / probe_spacing(n) - 0.5;
    let f = g - floor(g);
    return Bilinear(
        vec2<i32>(floor(g)),
        vec4(
            (1.0 - f.x) * (1.0 - f.y),
            f.x * (1.0 - f.y),
            (1.0 - f.x) * f.y,
            f.x * f.y,
        ),
    );
}

// Bilinear neighbor k, clamped into the canvas probe grid.
fn bilinear_probe(b: Bilinear, k: u32, n: u32) -> vec2<i32> {
    return clamp(b.base + BILINEAR_OFFSETS[k], vec2(0), vec2<i32>(probe_grid(n)) - 1);
}

fn interval(n: u32) -> vec2<f32> {
    let start = select(params.base_interval * f32(1u << (2u * n)), 0.0, n == 0u);
    let end = params.base_interval * f32(1u << (2u * n + 2u));
    return vec2(start, end);
}

fn ray_dir(ray: u32, count: u32) -> vec2<f32> {
    let angle = (f32(ray) + 0.5) * TAU / f32(count) + params.ray_rotation;
    return vec2(cos(angle), sin(angle));
}

fn load_ray(i: u32) -> vec4<f32> {
    let v = cascade_src[i];
    return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn store_ray(i: u32, r: vec4<f32>) {
    cascade_dst[i] = vec2(pack2x16float(r.xy), pack2x16float(r.zw));
}

fn merge(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    return vec4(near.rgb + near.a * far.rgb, near.a * far.a);
}

// Average of rays 4*ray .. 4*ray+3 of the probe in `slot` of cascade n: for a
// cascade n+1 probe, that's the arc of parent ray `ray`; for cascade 0 with
// ray = 0, it's the probe's whole fluence.
fn probe_arc(slot: u32, n: u32, ray: u32) -> vec4<f32> {
    let base = slot * ray_count(n) + ray * 4u;
    return 0.25 * (load_ray(base) + load_ray(base + 1u)
        + load_ray(base + 2u) + load_ray(base + 3u));
}

// Far field for one sub-direction: the child ray itself when merging per
// direction, else the pre-averaged arc of the parent ray.
fn child_value(slot: u32, n1: u32, parent_ray: u32, child_ray: u32, per_direction: bool) -> vec4<f32> {
    if per_direction {
        return load_ray(slot * ray_count(n1) + child_ray);
    }
    return probe_arc(slot, n1, parent_ray);
}

// ------------------------------------------------------ segments and hits
//
// Each stored ray of cascade n marches `sub_count(n) * segment_count(n)`
// segments, and the trace pass caches where each one first hit (as a
// fraction of the segment, unorm16; 1.0 = miss). The segments depend only
// on the probe's world position, its children's positions and the ray
// direction, never on lighting, so the cache stays valid until geometry or
// cascade settings change.
//
// Per-direction merge: a stored ray covers an arc, and tracing its near
// segment only along the arc's center makes near visibility all-or-nothing
// for the whole arc. Next to a bright surface that's biased bright:
// directions that hit are always counted (a child interval that starts
// inside the surface reports its emission), and a center hit also claims
// the arc's directions that miss. So cascades n < per_direction_levels
// trace each of the 4 child directions separately and average after the
// merge; an average of products isn't the product of averages.
//
// Bilinear fix: instead of one near segment merged with the interpolated
// far field (which leaks: the 4 child probes can sit on the far side of an
// occluder the near segment never saw), trace from this probe's interval
// start to where each child probe's interval starts, so each near segment
// actually connects to the far field it's merged with.

fn has_children(n: u32) -> bool {
    return n + 1u < params.cascade_count;
}

fn sub_count(n: u32) -> u32 {
    return select(1u, 4u, has_children(n) && n < params.per_direction_levels);
}

fn segment_count(n: u32) -> u32 {
    return select(1u, 4u, has_children(n) && params.merge_mode == 1u);
}

// u32 words per stored ray: two unorm16 hit fractions each.
fn hit_words(n: u32) -> u32 {
    return (sub_count(n) * segment_count(n) + 1u) / 2u;
}

struct Segment {
    a: vec2<f32>,
    b: vec2<f32>,
}

// Segment k of sub-direction j of `ray` from world probe `world`.
fn segment(n: u32, world: vec2<i32>, ray: u32, j: u32, k: u32) -> Segment {
    let center = (vec2<f32>(world) + 0.5) * probe_spacing(n);
    let range = interval(n);
    var dir = ray_dir(ray, ray_count(n));
    if sub_count(n) == 4u {
        dir = ray_dir(ray * 4u + j, ray_count(n + 1u));
    }
    let a = center + dir * range.x;
    if segment_count(n) == 4u {
        let child = bilinear_probe(bilinear(center, n + 1u), k, n + 1u);
        return Segment(a, (vec2<f32>(child) + 0.5) * probe_spacing(n + 1u) + dir * range.y);
    }
    return Segment(a, center + dir * range.y);
}

fn march_fraction(seg: Segment) -> f32 {
    let delta = seg.b - seg.a;
    let len = length(delta);
    let h = trace_hit(seg.a, delta / len, 0.0, len);
    if !h.hit {
        return 1.0;
    }
    // Keep hits strictly below the 1.0 miss code after unorm16 rounding.
    return min(distance(seg.a, h.pos) / len, 65534.0 / 65535.0);
}

fn hit_word(ray_slot: u32, n: u32, h: u32) -> u32 {
    return params.hit_base + ray_slot * hit_words(n) + h / 2u;
}

// Radiance and visibility along a cached segment. Lighting is evaluated
// here, at gather time: emission (the spot's cone uses the segment's
// direction) and, in bounce passes, bounce light at the hit.
fn cached_near(ray_slot: u32, n: u32, h: u32, seg: Segment) -> vec4<f32> {
    let fractions = unpack2x16unorm(hits[hit_word(ray_slot, n, h)]);
    let fraction = select(fractions.x, fractions.y, h % 2u == 1u);
    if fraction >= 1.0 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let pos = mix(seg.a, seg.b, fraction);
    let s = scene(pos);
    var radiance = s.emission * select(1.0, spot_factor(normalize(seg.b - seg.a)), s.spot > 0.0);
    if params.bounce_iteration > 0u && any(s.albedo > vec3(0.0)) {
        radiance += s.albedo * bounce_radiance(pos, false);
    }
    return vec4(radiance, 0.0);
}

// Re-march the rays of probes that entered the circle (or everything, after
// the CPU clears the tags). One thread per (slot, ray).
@compute @workgroup_size(64, 1, 1)
fn cascade_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.cascade;
    let rays = ray_count(n);
    let window = self_window();
    if id.x >= window.size * window.size * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(vec2(slot % window.size, slot / window.size), n, window.size);
    if !probe_valid(world, n, window.radius) || tags[window.tag_base + slot] == pack_tag(world) {
        return;
    }

    let segments = segment_count(n);
    let count = sub_count(n) * segments;
    for (var h = 0u; h < count; h += 2u) {
        let first = march_fraction(segment(n, world, ray, h / segments, h % segments));
        var second = 1.0;
        if h + 1u < count {
            second = march_fraction(segment(n, world, ray, (h + 1u) / segments, (h + 1u) % segments));
        }
        hits[hit_word(id.x, n, h)] = pack2x16unorm(vec2(first, second));
    }
    if ray == 0u {
        atomicAdd(&remarched[n], 1u);
    }
}

// After every cascade's trace: record which world probe each slot now holds.
@compute @workgroup_size(64, 1, 1)
fn cascade_commit(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.cascade;
    let window = self_window();
    if id.x >= window.size * window.size {
        return;
    }
    let world = slot_world(vec2(id.x % window.size, id.x / window.size), n, window.size);
    tags[window.tag_base + id.x] = select(0u, pack_tag(world), probe_valid(world, n, window.radius));
}

// Radiance for every valid (slot, ray) from its cached hits and the next
// cascade. Runs every frame, once per bounce pass.
@compute @workgroup_size(64, 1, 1)
fn cascade_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.cascade;
    let rays = ray_count(n);
    let window = self_window();
    if id.x >= window.size * window.size * rays {
        return;
    }
    let ray = id.x % rays;
    let slot = id.x / rays;
    let world = slot_world(vec2(slot % window.size, slot / window.size), n, window.size);
    if !probe_valid(world, n, window.radius) {
        store_ray(id.x, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    if !has_children(n) {
        store_ray(id.x, cached_near(id.x, n, 0u, segment(n, world, ray, 0u, 0u)));
        return;
    }

    // Children that aren't ready (outside their circle) drop out and the
    // rest are renormalized; with none, the far field is lost energy.
    let children = bilinear((vec2<f32>(world) + 0.5) * probe_spacing(n), n + 1u);
    let child = child_window();
    var weights = children.weights;
    var child_slots: array<u32, 4>;
    for (var k = 0u; k < 4u; k++) {
        let c = bilinear_probe(children, k, n + 1u);
        child_slots[k] = slot_index(world_slot(c, child.size), child.size);
        if !probe_ready(c, n + 1u, child) {
            weights[k] = 0.0;
        }
    }
    let total = dot(weights, vec4(1.0));
    weights = select(vec4(0.0), weights / total, total > 0.0);

    let per_direction = sub_count(n) == 4u;
    let segments = segment_count(n);
    var result = vec4(0.0);
    for (var j = 0u; j < sub_count(n); j++) {
        let child_ray = ray * 4u + j;
        if segments == 1u {
            let near = cached_near(id.x, n, j, segment(n, world, ray, j, 0u));
            var far = vec4(0.0, 0.0, 0.0, select(1.0, 0.0, total > 0.0));
            for (var k = 0u; k < 4u; k++) {
                if weights[k] > 0.0 {
                    far += weights[k] * child_value(child_slots[k], n + 1u, ray, child_ray, per_direction);
                }
            }
            result += merge(near, far);
        } else if total > 0.0 {
            for (var k = 0u; k < 4u; k++) {
                if weights[k] > 0.0 {
                    let near = cached_near(id.x, n, j * 4u + k, segment(n, world, ray, j, k));
                    result += weights[k] * merge(near, child_value(child_slots[k], n + 1u, ray, child_ray, per_direction));
                }
            }
        } else {
            result += cached_near(id.x, n, j * 4u, segment(n, world, ray, j, 0u));
        }
    }
    store_ray(id.x, result / f32(sub_count(n)));
}

// ------------------------------------------------------------ reference

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Radiance arriving along one ray, path traced through up to `bounces`
// diffuse bounces with cosine-weighted directions (pdf cos/2 in 2D), so each
// bounce's estimator is albedo * L_in, matching bounce_radiance.
fn path_radiance(start: vec2<f32>, start_dir: vec2<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var origin = start;
    var dir = start_dir;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);
    for (var bounce = 0u; bounce <= params.bounces; bounce++) {
        let h = trace_hit(origin, dir, 0.0, 1e6);
        if !h.hit {
            break;
        }
        radiance += throughput * h.emission;
        if bounce == params.bounces || all(h.albedo == vec3(0.0)) {
            break;
        }
        throughput *= h.albedo;
        let n = scene_normal(h.pos);
        *rng = pcg(*rng);
        let theta = asin(2.0 * f32(*rng) / 4294967296.0 - 1.0);
        dir = cos(theta) * n + sin(theta) * vec2(-n.y, n.x);
        // HIT_EPS is 0.05; start clear of the surface.
        origin = h.pos + n * 0.1;
    }
    return radiance;
}

// Per-pixel fluence by brute force: stratified angles with a per-pixel,
// per-frame random rotation, so it converges to the continuous integral
// rather than to any fixed set of directions the cascades might share.
@compute @workgroup_size(8, 8, 1)
fn reference_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) {
        return;
    }
    let i = id.y * params.size.x + id.x;
    let center = vec2<f32>(id.xy) + 0.5;
    let jitter = f32(pcg(id.x + pcg(id.y + pcg(params.ref_frame)))) / 4294967296.0;

    var rng = pcg(id.x ^ pcg(id.y ^ pcg(params.ref_frame + 0x9e3779b9u)));

    // Wall pixels: the lit color of the nearest face, i.e. albedo times
    // cosine-weighted incoming light there. Cosine-distributed stratified
    // directions (u stratified, theta = asin(2u - 1)) make that estimator
    // albedo * L_in, like the cascades' bounce_radiance.
    let here = scene(center);
    let on_wall = here.dist <= 0.0 && any(here.albedo > vec3(0.0));
    let face = nearest_face(center, here.dist);

    var sum = vec3(0.0);
    for (var k = 0u; k < params.ref_rays; k++) {
        let u = (f32(k) + jitter) / f32(params.ref_rays);
        if on_wall {
            let theta = asin(2.0 * u - 1.0);
            let n = face.normal;
            let dir = cos(theta) * n + sin(theta) * vec2(-n.y, n.x);
            // HIT_EPS is 0.05; start clear of the surface.
            sum += here.albedo * path_radiance(face.pos + n * 0.1, dir, &rng);
        } else {
            let angle = u * TAU;
            sum += path_radiance(center, vec2(cos(angle), sin(angle)), &rng);
        }
    }

    let previous = select(reference[i], vec4(0.0), params.ref_frame == 0u);
    reference[i] = previous + vec4(sum, f32(params.ref_rays));
}

// -------------------------------------------------------------- resolve

fn tonemap(v: vec3<f32>) -> vec3<f32> {
    return 1.0 - exp(-v * params.exposure);
}

// Reads cascade 0 from cascade_src.
@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) {
        return;
    }
    let i = id.y * params.size.x + id.x;

    // Bilinear over the 4 nearest cascade-0 probes. At s0 = 1 the pixel
    // center is exactly on its own probe, so this is just that probe.
    // Probes centered inside a shape report that shape's emission, not the
    // light field around it; blending them in paints a halo around every
    // shape edge, so they're dropped and the remaining weights renormalized.
    // (A pixel inside a shape keeps them all: it wants the emission.)
    // Only pixels inside the circle of live probes are resolved and
    // measured; outside, the reference is shown dimmed for context.
    let pixel_center = vec2<f32>(id.xy) + 0.5;
    let circle_distance = distance(pixel_center, params.circle_center);
    let in_circle = circle_distance <= params.c0_radius;
    let here = scene(pixel_center);
    let pixel_empty = here.dist > 0.0;
    let on_wall = in_circle && !pixel_empty && any(here.albedo > vec3(0.0));
    let probes = bilinear(pixel_center, 0u);
    let window = c0_window();
    var rc = vec3(0.0);
    var rc_all = vec3(0.0);
    var weight = 0.0;
    var weight_all = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let probe = bilinear_probe(probes, k, 0u);
        if !probe_ready(probe, 0u, window) {
            continue;
        }
        let value = probe_arc(slot_index(world_slot(probe, window.size), window.size), 0u, 0u).rgb;
        rc_all += probes.weights[k] * value;
        weight_all += probes.weights[k];
        let probe_center = (vec2<f32>(probe) + 0.5) * probe_spacing(0u);
        if !pixel_empty || scene(probe_center).dist > 0.0 {
            rc += probes.weights[k] * value;
            weight += probes.weights[k];
        }
    }
    // All 4 inside shapes (a gap thinner than the probe spacing): nothing
    // better to go on than the plain blend.
    rc = select(select(vec3(0.0), rc_all / weight_all, weight_all > 0.0), rc / weight, weight > 0.0);
    // Wall pixels show their nearest face's lit color from this pass's
    // cascade 0, looked up from that face's side only, like the bounces.
    if on_wall {
        let face = nearest_face(pixel_center, here.dist);
        rc = here.albedo * bounce_radiance(face.pos, true);
    }
    if params.rc_accumulate == 1u {
        let history = select(rc_history[i], vec4(0.0), params.rc_frame == 0u) + vec4(rc, 1.0);
        rc_history[i] = history;
        rc = history.rgb / history.a;
    }
    let acc = reference[i];
    let ref_value = acc.rgb / max(acc.a, 1.0);
    let e = rc - ref_value;

    // The main metric covers empty space only; inside lights both sides just
    // report the emission. Wall faces are flagged for their own metric.
    error[2u * i] = vec4(
        dot(e, e),
        dot(ref_value, ref_value),
        e.r + e.g + e.b,
        select(-acc.a, acc.a, pixel_empty && in_circle),
    );
    error[2u * i + 1u] = vec4(
        dot(ref_value, vec3(1.0)),
        dot(rc, vec3(1.0)),
        select(0.0, 1.0, on_wall),
        0.0,
    );

    var shown: vec3<f32>;
    switch params.view_mode {
        case 0u: { shown = tonemap(rc); }
        case 1u: { shown = tonemap(ref_value); }
        default: { shown = saturate(abs(e) * params.exposure * params.diff_gain); }
    }
    if !in_circle {
        shown = 0.3 * tonemap(ref_value);
    }
    if abs(circle_distance - params.c0_radius) < 0.75 {
        shown = vec3(1.0);
    }
    textureStore(display, id.xy, vec4(shown, 1.0));
}
