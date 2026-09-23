// 2D radiance cascades testbed, plus a brute-force reference over the
// exact same scene and trace function, so any difference between the two
// is the cascade approximation and nothing else.
//
// Conventions (checked against m4xc.dev "Fundamentals of Radiance Cascades"
// and Osborne & Sannikov 2024, arXiv:2408.14425):
// - Branching factor 4: cascade n has 4^(n+1) rays per probe, probe spacing
//   2^n px (s0 = 1 px, one cascade-0 probe per pixel), interval
//   [b*4^n, b*4^(n+1)] with cascade 0 starting at 0. Memory per cascade is
//   constant: (W/2^n)(H/2^n) * 4^(n+1) = 4WH.
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
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cascade_src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cascade_dst: array<vec4<f32>>;
// rgb = summed radiance, a = number of rays summed.
@group(0) @binding(3) var<storage, read_write> reference: array<vec4<f32>>;
// Per pixel: (squared rgb error, squared reference, empty-space mask,
// reference ray count). Read back on the CPU for the relative RMSE.
@group(0) @binding(4) var<storage, read_write> error: array<vec4<f32>>;
@group(0) @binding(5) var display: texture_storage_2d<rgba8unorm, write>;

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
    s = closer(s, sd_circle(p, vec2(110.0, 400.0), 22.0), vec3(8.0, 4.5, 1.5));
    s = closer(s, sd_circle(p, vec2(420.0, 100.0), 14.0), vec3(1.5, 3.5, 10.0));
    s = closer(s, sd_box(p, vec2(400.0, 490.0), vec2(80.0, 3.0)), vec3(0.4, 3.0, 0.8));
    // Thin wall right of the warm light: the leak test. Anything warm to
    // its right is light that got interpolated through it.
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
    return f32(1u << n);
}

fn probe_grid(n: u32) -> vec2<u32> {
    return params.size >> vec2(n);
}

fn interval(n: u32) -> vec2<f32> {
    let start = select(params.base_interval * f32(1u << (2u * n)), 0.0, n == 0u);
    let end = params.base_interval * f32(1u << (2u * n + 2u));
    return vec2(start, end);
}

fn ray_dir(ray: u32, count: u32) -> vec2<f32> {
    let angle = (f32(ray) + 0.5) * TAU / f32(count);
    return vec2(cos(angle), sin(angle));
}

fn merge(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    return vec4(near.rgb + near.a * far.rgb, near.a * far.a);
}

// Cascade n+1 radiance for parent ray `ray` of cascade n at child probe
// `probe`: the average of the 4 child rays spanning the parent's arc.
fn child_radiance(probe: vec2<u32>, ray: u32) -> vec4<f32> {
    let n1 = params.cascade + 1u;
    let base = (probe.y * probe_grid(n1).x + probe.x) * ray_count(n1) + ray * 4u;
    return 0.25 * (cascade_src[base] + cascade_src[base + 1u]
        + cascade_src[base + 2u] + cascade_src[base + 3u]);
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
        cascade_dst[id.x] = trace(center, dir, range.x, range.y);
        return;
    }

    let child_spacing = probe_spacing(n + 1u);
    let child_grid = vec2<i32>(probe_grid(n + 1u));
    let g = center / child_spacing - 0.5;
    let base = vec2<i32>(floor(g));
    let f = g - floor(g);
    let offsets = array(vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1));
    let weights = array(
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y,
    );

    var result = vec4(0.0);
    if params.merge_mode == 0u {
        // Plain: one near segment, merged with the interpolated far field.
        // Leaks: the 4 child probes can sit on the far side of an occluder
        // the near segment never saw.
        let near = trace(center, dir, range.x, range.y);
        var far = vec4(0.0);
        for (var k = 0; k < 4; k++) {
            let c = vec2<u32>(clamp(base + offsets[k], vec2(0), child_grid - 1));
            far += weights[k] * child_radiance(c, ray);
        }
        result = merge(near, far);
    } else {
        // Bilinear fix: trace from this probe's interval start to where each
        // child probe's interval starts, so each near segment actually
        // connects to the far field it's merged with.
        let start = center + dir * range.x;
        for (var k = 0; k < 4; k++) {
            let c = vec2<u32>(clamp(base + offsets[k], vec2(0), child_grid - 1));
            let child_start = (vec2<f32>(c) + 0.5) * child_spacing + dir * range.y;
            let near = trace_segment(start, child_start);
            result += weights[k] * merge(near, child_radiance(c, ray));
        }
    }
    cascade_dst[id.x] = result;
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

    // s0 = 1 px: this pixel's own cascade-0 probe, centered on it.
    let rc = 0.25 * (cascade_src[i * 4u] + cascade_src[i * 4u + 1u]
        + cascade_src[i * 4u + 2u] + cascade_src[i * 4u + 3u]).rgb;
    let acc = reference[i];
    let ref_value = acc.rgb / max(acc.a, 1.0);
    let e = rc - ref_value;

    // Error is only meaningful in empty space; inside shapes both sides
    // just report the shape's own emission.
    let empty = scene(vec2<f32>(id.xy) + 0.5).dist > 0.0;
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
