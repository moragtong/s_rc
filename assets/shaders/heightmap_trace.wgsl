#define_import_path s_rc::heightmap_trace

// Shared by the 2.5D cascade passes and the shading pass: the dual
// heightmap, the cascade uniforms, ray marching through the heightmap,
// and probe placement. Must mirror src/cascades2_5d.

struct CascadeInfo {
    size: vec2<u32>,
    interval: vec2<f32>,
    // First ray of the cascade in `rays`.
    ray_base: u32,
    // First probe of the cascade in `sh_probes`.
    probe_base: u32,
    spacing: f32,
    octa_side: u32,
    // SH projection sub-samples per axis per cell, and the cascade's first.
    sub: u32,
    sub_base: u32,
    // Uniform arrays need a 16-byte stride.
    pad_a: u32,
    pad_b: u32,
}

struct Uniforms {
    hmap_min: vec2<f32>,
    hmap_max: vec2<f32>,
    hmap_res: vec2<u32>,
    cascade_count: u32,
    // Highest surface in the map; rays climbing above it can't hit.
    hmap_top: f32,
    // Unit vector toward the sun.
    sun_toward: vec3<f32>,
    // Sun irradiance, already multiplied by the camera exposure, so every
    // radiance here is in the same units as the main pass's output.
    sun_irradiance: vec3<f32>,
    cascades: array<CascadeInfo, 4>,
}

// x: ceiling (lowest surface above the split), y: floor (highest below).
@group(0) @binding(0) var heightmap: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

const PI: f32 = 3.14159265;
// Stored instead of a distance for rays that hit nothing.
const MISS: f32 = -1.0;
// SH probes: 9 coefficients, then (position, parallax radius).
const SH_STRIDE: u32 = 10u;
// Probes sit midway between floor and ceiling, but at least this far above
// the floor and at most PROBE_MAX_LIFT (open-sky columns have no ceiling).
const PROBE_MIN_LIFT: f32 = 0.1;
const PROBE_MAX_LIFT: f32 = 2.0;

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    if n.x >= 0.0 { n.x -= t; } else { n.x += t; }
    if n.y >= 0.0 { n.y -= t; } else { n.y += t; }
    return normalize(n);
}

// Center direction of cell `i` of a side x side octahedral map.
fn cell_direction(i: u32, side: u32) -> vec3<f32> {
    let uv = (vec2<f32>(f32(i % side), f32(i / side)) + 0.5) / f32(side);
    return oct_decode(uv);
}

fn sh_basis(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        -0.488603 * d.y,
        0.488603 * d.z,
        -0.488603 * d.x,
        1.092548 * d.x * d.y,
        -1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        -1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

fn cell_size() -> vec2<f32> {
    return (uniforms.hmap_max - uniforms.hmap_min) / vec2<f32>(uniforms.hmap_res);
}

fn texel_at(xz: vec2<f32>) -> vec2<i32> {
    let t = vec2<i32>(floor((xz - uniforms.hmap_min) / cell_size()));
    return clamp(t, vec2<i32>(0), vec2<i32>(uniforms.hmap_res) - 1);
}

fn probe_position(c: u32, probe: vec2<u32>) -> vec3<f32> {
    let s = uniforms.cascades[c].spacing;
    let xz = (vec2<f32>(probe) + 0.5) * s + uniforms.hmap_min;
    let h = textureLoad(heightmap, texel_at(xz), 0).xy;
    let lift = clamp(0.5 * (h.x - h.y), PROBE_MIN_LIFT, PROBE_MAX_LIFT);
    return vec3<f32>(xz.x, h.y + lift, xz.y);
}

struct Hit {
    // Distance along the ray, MISS if none.
    t: f32,
    texel: vec2<i32>,
    // Hit the ceiling surface (or the side of a ceiling column), not the
    // floor.
    ceiling: bool,
}

fn nonzero(v: vec2<f32>) -> vec2<f32> {
    let tiny = select(vec2<f32>(-1e-8), vec2<f32>(1e-8), v >= vec2<f32>(0.0));
    return select(v, tiny, abs(v) < vec2<f32>(1e-8));
}

// First solid point of the ray o + t d, t in [t_start, t_end]; `d` is unit.
// A column is solid below its floor and above its ceiling. Outside the
// map's XZ bounds is empty.
fn trace(o: vec3<f32>, d: vec3<f32>, t_start: f32, t_end: f32) -> Hit {
    var hit: Hit;
    hit.t = MISS;

    let inv = 1.0 / nonzero(d.xz);
    let ta = (uniforms.hmap_min - o.xz) * inv;
    let tb = (uniforms.hmap_max - o.xz) * inv;
    let t_near = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), t_start);
    let t_far = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), t_end);
    if t_near >= t_far {
        return hit;
    }

    let cell = cell_size();
    var texel = texel_at(o.xz + d.xz * t_near);
    let dir_step = select(vec2<i32>(-1), vec2<i32>(1), d.xz >= vec2<f32>(0.0));
    let next_edge = uniforms.hmap_min
        + (vec2<f32>(texel) + select(vec2<f32>(0.0), vec2<f32>(1.0), dir_step > vec2<i32>(0))) * cell;
    var t_max = (next_edge - o.xz) * inv;
    let t_delta = abs(cell * inv);

    var t = t_near;
    for (var i = 0u; i < 2048u; i++) {
        let t_out = min(min(t_max.x, t_max.y), t_far);
        let h = textureLoad(heightmap, texel, 0).xy;
        let y_in = o.y + d.y * t;
        hit.texel = texel;

        // Entering a column below its floor or above its ceiling: the side
        // of a step or a beam.
        if y_in <= h.y {
            hit.t = t;
            hit.ceiling = false;
            return hit;
        }
        if y_in >= h.x {
            hit.t = t;
            hit.ceiling = true;
            return hit;
        }
        if d.y < 0.0 {
            let t_hit = (h.y - o.y) / d.y;
            if t_hit <= t_out {
                hit.t = t_hit;
                hit.ceiling = false;
                return hit;
            }
        } else if d.y > 0.0 {
            if y_in > uniforms.hmap_top {
                break;
            }
            let t_hit = (h.x - o.y) / d.y;
            if t_hit <= t_out {
                hit.t = t_hit;
                hit.ceiling = true;
                return hit;
            }
        }
        if t_out >= t_far {
            break;
        }
        if t_max.x < t_max.y {
            t = t_max.x;
            t_max.x += t_delta.x;
            texel.x += dir_step.x;
        } else {
            t = t_max.y;
            t_max.y += t_delta.y;
            texel.y += dir_step.y;
        }
        if any(texel < vec2<i32>(0)) || any(texel >= vec2<i32>(uniforms.hmap_res)) {
            break;
        }
    }
    hit.t = MISS;
    return hit;
}
