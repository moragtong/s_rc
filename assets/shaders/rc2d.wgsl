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
}

@group(0) @binding(0) var<uniform> params: Params;
// Cascade entries are rgba16float packed into two u32s (pack2x16float), so
// half the bandwidth of vec4<f32> without needing the f16 shader feature.
@group(0) @binding(1) var<storage, read> cascade_src: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> cascade_dst: array<vec2<u32>>;
// rgb = summed radiance, a = number of rays summed.
@group(0) @binding(3) var<storage, read_write> reference: array<vec4<f32>>;
// Per pixel: (squared rgb error, squared reference, empty-space mask,
// reference ray count). Read back on the CPU for the relative RMSE.
@group(0) @binding(4) var<storage, read_write> error: array<vec4<f32>>;
@group(0) @binding(5) var display: texture_storage_2d<rgba8unorm, write>;
// rgb = summed resolved cascades, a = frame count (rc_accumulate only).
@group(0) @binding(6) var<storage, read_write> rc_history: array<vec4<f32>>;

const TAU: f32 = 6.28318530718;
const HIT_EPS: f32 = 0.05;
const MAX_STEPS: u32 = 256u;

// ---------------------------------------------------------------- scene

struct SceneSample {
    dist: f32,
    emission: vec3<f32>,
}

fn sd_circle(p: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn sd_box(p: vec2<f32>, center: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let q = abs(p - center) - half_size;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}

fn closer(a: SceneSample, dist: f32, emission: vec3<f32>) -> SceneSample {
    if dist < a.dist {
        return SceneSample(dist, emission);
    }
    return a;
}

// Pixel units, y down. Emitters also occlude; zero emission = black occluder.
fn scene(p: vec2<f32>) -> SceneSample {
    var s = SceneSample(1e9, vec3(0.0));
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

// Radiance along [t0, t1] of a ray; a = 1 if it hit nothing. Rays that leave
// the canvas never re-enter it (it's convex), so they miss.
fn trace(origin: vec2<f32>, dir: vec2<f32>, t0: f32, t1: f32) -> vec4<f32> {
    let bounds = vec2<f32>(params.size);
    var t = t0;
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = origin + dir * t;
        if any(p < vec2(0.0)) || any(p > bounds) {
            break;
        }
        let s = scene(p);
        if s.dist < HIT_EPS {
            return vec4(s.emission, 0.0);
        }
        t += s.dist;
        if t >= t1 {
            break;
        }
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
}

fn trace_segment(a: vec2<f32>, b: vec2<f32>) -> vec4<f32> {
    let seg = b - a;
    let len = length(seg);
    return trace(a, seg / len, 0.0, len);
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

// Cell-centered bilinear footprint of cascade n's probes at `pos`.
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

fn bilinear_probe(b: Bilinear, k: u32, n: u32) -> vec2<u32> {
    return vec2<u32>(clamp(b.base + BILINEAR_OFFSETS[k], vec2(0), vec2<i32>(probe_grid(n)) - 1));
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

// Average of rays 4*ray .. 4*ray+3 of `probe` in cascade n: for a cascade
// n+1 probe, that's the arc of parent ray `ray`; for cascade 0 with
// ray = 0, it's the probe's whole fluence.
fn probe_arc(probe: vec2<u32>, n: u32, ray: u32) -> vec4<f32> {
    let base = (probe.y * probe_grid(n).x + probe.x) * ray_count(n) + ray * 4u;
    return 0.25 * (load_ray(base) + load_ray(base + 1u)
        + load_ray(base + 2u) + load_ray(base + 3u));
}

fn probe_ray(probe: vec2<u32>, n: u32, ray: u32) -> vec4<f32> {
    return load_ray((probe.y * probe_grid(n).x + probe.x) * ray_count(n) + ray);
}

// Far field for one sub-direction: the child ray itself when merging per
// direction, else the pre-averaged arc of the parent ray.
fn child_value(probe: vec2<u32>, n1: u32, parent_ray: u32, child_ray: u32, per_direction: bool) -> vec4<f32> {
    if per_direction {
        return probe_ray(probe, n1, child_ray);
    }
    return probe_arc(probe, n1, parent_ray);
}

@compute @workgroup_size(64, 1, 1)
fn cascade(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.cascade;
    let rays = ray_count(n);
    let grid = probe_grid(n);
    if id.x >= grid.x * grid.y * rays {
        return;
    }

    let ray = id.x % rays;
    let probe_index = id.x / rays;
    let probe = vec2(probe_index % grid.x, probe_index / grid.x);
    let center = (vec2<f32>(probe) + 0.5) * probe_spacing(n);
    let dir = ray_dir(ray, rays);
    let range = interval(n);

    if n + 1u >= params.cascade_count {
        store_ray(id.x, trace(center, dir, range.x, range.y));
        return;
    }

    let children = bilinear(center, n + 1u);
    let child_rays = ray_count(n + 1u);

    // Per-direction merge: a stored ray covers an arc, and tracing its near
    // segment only along the arc's center makes near visibility all-or-
    // nothing for the whole arc. Next to a bright surface that's biased
    // bright: directions that hit are always counted (a child interval that
    // starts inside the surface reports its emission), and a center hit
    // also claims the arc's directions that miss. Tracing each child
    // direction separately and averaging after the merge removes that; an
    // average of products isn't the product of averages.
    let per_direction = n < params.per_direction_levels;
    let sub_count = select(1u, 4u, per_direction);
    var result = vec4(0.0);
    for (var j = 0u; j < sub_count; j++) {
        let child_ray = ray * 4u + j;
        let sub_dir = select(dir, ray_dir(child_ray, child_rays), per_direction);
        if params.merge_mode == 0u {
            // Plain: one near segment, merged with the interpolated far
            // field. Leaks: the 4 child probes can sit on the far side of an
            // occluder the near segment never saw.
            let near = trace(center, sub_dir, range.x, range.y);
            var far = vec4(0.0);
            for (var k = 0u; k < 4u; k++) {
                let c = bilinear_probe(children, k, n + 1u);
                far += children.weights[k] * child_value(c, n + 1u, ray, child_ray, per_direction);
            }
            result += merge(near, far);
        } else {
            // Bilinear fix: trace from this probe's interval start to where
            // each child probe's interval starts, so each near segment
            // actually connects to the far field it's merged with.
            let start = center + sub_dir * range.x;
            for (var k = 0u; k < 4u; k++) {
                let c = bilinear_probe(children, k, n + 1u);
                let child_start = (vec2<f32>(c) + 0.5) * probe_spacing(n + 1u) + sub_dir * range.y;
                let near = trace_segment(start, child_start);
                result += children.weights[k] * merge(near, child_value(c, n + 1u, ray, child_ray, per_direction));
            }
        }
    }
    result /= f32(sub_count);
    store_ray(id.x, result);
}

// ------------------------------------------------------------ reference

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
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

    var sum = vec3(0.0);
    for (var k = 0u; k < params.ref_rays; k++) {
        let angle = (f32(k) + jitter) * TAU / f32(params.ref_rays);
        sum += trace(center, vec2(cos(angle), sin(angle)), 0.0, 1e6).rgb;
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
    let pixel_center = vec2<f32>(id.xy) + 0.5;
    let pixel_empty = scene(pixel_center).dist > 0.0;
    let probes = bilinear(pixel_center, 0u);
    var rc = vec3(0.0);
    var rc_all = vec3(0.0);
    var weight = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let probe = bilinear_probe(probes, k, 0u);
        let value = probe_arc(probe, 0u, 0u).rgb;
        rc_all += probes.weights[k] * value;
        let probe_center = (vec2<f32>(probe) + 0.5) * probe_spacing(0u);
        if !pixel_empty || scene(probe_center).dist > 0.0 {
            rc += probes.weights[k] * value;
            weight += probes.weights[k];
        }
    }
    // All 4 inside shapes (a gap thinner than the probe spacing): nothing
    // better to go on than the plain blend.
    rc = select(rc_all, rc / weight, weight > 0.0);
    if params.rc_accumulate == 1u {
        let history = select(rc_history[i], vec4(0.0), params.rc_frame == 0u) + vec4(rc, 1.0);
        rc_history[i] = history;
        rc = history.rgb / history.a;
    }
    let acc = reference[i];
    let ref_value = acc.rgb / max(acc.a, 1.0);
    let e = rc - ref_value;

    // Error is only meaningful in empty space; inside shapes both sides
    // just report the shape's own emission.
    let empty = pixel_empty;
    error[i] = select(
        vec4(0.0, 0.0, 0.0, acc.a),
        vec4(dot(e, e), dot(ref_value, ref_value), 1.0, acc.a),
        empty,
    );

    var shown: vec3<f32>;
    switch params.view_mode {
        case 0u: { shown = tonemap(rc); }
        case 1u: { shown = tonemap(ref_value); }
        default: { shown = saturate(abs(e) * params.exposure * params.diff_gain); }
    }
    textureStore(display, id.xy, vec4(shown, 1.0));
}
