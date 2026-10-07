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
    // The bake plane (SPLIT): the height probes sit at now that the lift is
    // disabled.
    split: f32,
    // Unit vector toward the sun.
    sun_toward: vec3<f32>,
    // Sun irradiance, already multiplied by the camera exposure, so every
    // radiance here is in the same units as the main pass's output.
    sun_irradiance: vec3<f32>,
    // Radiance a probe ray stores when it hits nothing: the open sky. Enters
    // the same bilinear merge as surface radiance, no special casing.
    sky_color: vec4<f32>,
    cascades: array<CascadeInfo, 4>,
    // 1 to end a child trace's segments at the nearest projected parent hit,
    // 0 to hand over to the old `q + d * parent.interval` endpoint. Only the
    // segment ends move; everything else in the parent/grid plumbing is the
    // same either way.
    parallax_endpoint: u32,
    parallax_pad: u32,
}

// x: ceiling (lowest surface above the split), y: floor (highest below).
@group(0) @binding(0) var heightmap: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
// Surface normals of the two channels. "min"/"max" name the height texture's
// two channels, not low and high: h.x (ceiling) pairs with the ceiling normal
// and h.y (floor) with the floor normal, same as src/heightmap_bake.rs.
@group(0) @binding(2) var ceiling_normal: texture_2d<f32>;
@group(0) @binding(3) var floor_normal: texture_2d<f32>;
// 2D winding number of the mesh cut at SPLIT, one texel per heightmap texel
// and with the same bounds, so `texel` indexes it directly. The horizontal
// containment signal; see `is_solid`.
@group(0) @binding(4) var winding: texture_2d<f32>;

const PI: f32 = 3.14159265;
// Stored instead of a distance for rays that hit nothing.
const MISS: f32 = -1.0;
// SH probes: 9 coefficients, then (position, parallax radius).
const SH_STRIDE: u32 = 10u;
// Probes sit midway between floor and ceiling, but at least this far above
// the floor and at most PROBE_MAX_LIFT (open-sky columns have no ceiling).
const PROBE_MIN_LIFT: f32 = 0.1;
const PROBE_MAX_LIFT: f32 = 2.0;
// How far clear of a surface a relocated probe is left. Small enough to stay
// inside the open column, large enough that the surface it just left does not
// immediately re-embed it.
const PROBE_SURFACE_EPS: f32 = 0.02;
// Smallest ring a buried probe is moved out over, m. The storey slab that
// makes up almost every flagged column on Sponza is 0.76 m thick, so a move
// shorter than this leaves the probe inside it; the clearance a cascade needs
// on top of this is RELOCATE's business, not the ring's.
const RELOCATE_MIN_RADIUS: f32 = 0.75;
// Floor on |n.y| when turning a vertical push into a distance along the
// surface normal. A vertical wall has n.y near zero, and dividing by it would
// turn a few centimetres of overshoot into a huge sideways leap.
const PROBE_NORMAL_MIN_Y: f32 = 0.2;

// Height tolerance between a probe's own height and the shaded pixel's, as a
// multiple of the cascade's spacing. Cascade-0 spacing is 1.0 m, so this is a
// 3.0 m tolerance - wider than the 2.7 m from the probe plane to the floor or
// the ceiling, which is what the room needs: every probe sits at
// `uniforms.split` (see `probe_position`), so a pixel on the floor or the
// ceiling is a full 2.7 m away from every probe that can light it.
//
// At 2.0 the cutoff fell inside the room and the whole floor and ceiling lost
// indirect light: 47.8% of the scene's pixels came back with zero contribution
// and the top and bottom of the frame read as flat slices. At 3.0 that is 0.6%
// and no band edges survive. The lower bound on the tolerance is therefore
// the height of the space, not the cascade's cell size.
//
// The cutoff itself is unchanged: past the tolerance a probe contributes
// nothing at all rather than fading, because a probe on another storey - or
// inside a wall whose column floor is the ground floor - must not keep a small
// nonzero share when the four probes of a footprint renormalise against each
// other. This is a cutoff, not a falloff; only its width moved.
const HEIGHT_TOLERANCE_SPACING: f32 = 3.0;

// Fraction of the tolerance over which the cutoff fades, to avoid aliasing.
const HEIGHT_CUTOFF_SHOULDER: f32 = 0.85;

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
    let info = uniforms.cascades[c];
    let s = info.spacing;
    let xz = (vec2<f32>(probe) + 0.5) * s + uniforms.hmap_min;
    // Every probe sits on the bake plane now: no relocation out of a buried
    // column, no lift off the floor, no burial push. The lift and the escape
    // search both read surfaces off a floor/ceiling map, which is the
    // machinery this experiment is removing; `escape_texel`, `lifted_height`
    // and the constants they use stay in the module, unreferenced, so the
    // original placement is one function's worth of code away.
    return vec3<f32>(xz.x, uniforms.split, xz.y);
}

// One of the segments `march_hits.wgsl` stored for a ray: where it started and
// which way it ran (not necessarily unit - `trace` was given `dir / length(dir)`,
// so `t` is measured from `start` along the normalised direction).
struct Segment {
    start: vec3<f32>,
    dir: vec3<f32>,
}

// The segment cascade `cc` stored for probe `qi`, direction cell `pr`, slot `k`.
// The march writes these and `cascade()` reads them back, so both derive the
// geometry here rather than in parallel: the top cascade's own trace runs
// straight out from `q`, and every other cascade's four segments run from the
// ray's interval start to where its own parent's rays of the same direction
// start - the same reconstruction `march_hits` did when it traced them.
//
// `pr` is both the direction the segment runs in and the direction whose
// radiance `rays` holds for it, which is what lets the merge read the parent
// back in the direction that was actually intersected - see
// `parent_dir_hits` in `cascades2_5d.wgsl`.
fn stored_segment(cc: u32, qi: vec2<u32>, pr: u32, k: u32) -> Segment {
    let info = uniforms.cascades[cc];
    let q = probe_position(cc, qi);
    let d = cell_direction(pr, info.octa_side);
    var seg: Segment;
    seg.start = q;
    seg.dir = d;
    if cc + 1u >= uniforms.cascade_count {
        return seg;
    }
    let grand = uniforms.cascades[cc + 1u];
    seg.start = q + d * info.interval.x;
    let u = (q.xz - uniforms.hmap_min) / grand.spacing - 0.5;
    let cell = vec2<i32>(floor(u));
    let corner = vec2<i32>(i32(k & 1u), i32(k >> 1u));
    let qi2 = vec2<u32>(clamp(cell + corner, vec2<i32>(0), vec2<i32>(grand.size) - 1));
    seg.dir = probe_position(cc + 1u, qi2) + d * grand.interval.x - seg.start;
    return seg;
}

// The grid is top-down but each probe is lifted off its own column's floor, so
// the four probes of a footprint sit at four unrelated heights - and 14% of
// cascade-0 footprints straddle a storey by more than 2 m. An XZ bilinear
// footprint cannot express that: a vertical offset leaves XZ untouched, so
// reprojecting a probe "to the pixel's height" returns the same four cells with
// the same weights. The height difference has to enter as a weight, or a probe
// on another storey - or inside a hollow in a wall, whose column floor is the
// ground floor - collects a full bilinear share and leaks.
//
// `spacing` scales the tolerance so it tracks cascade coarseness.
//
// Hard cutoff, not a falloff: past the tolerance the probe contributes nothing
// at all. A Gaussian never reaches zero, so a probe on another storey kept
// collecting a small but nonzero share, and since the four probes of a
// footprint are renormalised against each other, that residual share was
// enough to leak. The soft shoulder only exists to keep the transition from
// aliasing as the pixel moves; it spans a small fraction of the tolerance.
// How far a stored normal must lean *away* from the open column before its
// column counts as buried. A surface in open air always faces the ray that
// found it: a floor from above, a ceiling from below. Only a column that starts
// inside material records the far side of it.
const BURIED_MIN_Y: f32 = -0.5;

// How far around a column the split cut has to wind before it counts as
// material. A closed loop gives exactly 1 inside and 0 outside, so 0.5 is the
// midpoint; Sponza's shells are open, so the sum is fractional and the
// reference implementation was measured against this value rather than
// assumed: 98.5% of the texels the normals flag, 0.0% of clearly open ones.
const WINDING_SOLID: f32 = 0.5;

// Whether the column at `texel` is inside solid material at the split.
//
// Decided by the winding number alone. A floor and a ceiling cannot say "this
// column is inside a wall" - they only say where the air is - which is what let
// probes form inside masonry and take a bilinear share from both sides of it.
// The winding cut is a horizontal slice through the same geometry that a wall
// crosses head-on, so it is the term that is actually about a wall.
//
// The two stored normals used to be the first term. They are a backface
// signal: the bake cameras rasterise backfaces, so a column that starts in
// open air records the surface its ray lands on while one that starts inside a
// wall records the surface the ray *leaves* through, and both of those face
// away from the split - hence `BURIED_MIN_Y` on either channel. That works,
// but it is a statement about the floor and the ceiling of a column, not about
// what is between them, and it is blind to a vertical surface in exactly the
// way the heightmap itself is: the bake's rays run only up and down, so both
// sides of a wall record open air. The previous form is kept in the body,
// commented, with the reasoning that went with it.
//
// It was an addition, not a replacement, on Sponza: there it reproduced 98.5%
// of what the normals flagged and added columns they could not reach (14187
// texels, of which 11922 were new).
//
// Measured on `room.glb` with the two channels dumped (`S_RC_DUMP_META_MIN` /
// `_MAX`, 512^2): the normals are *not* dead here - 2520 ceiling and 2328
// floor texels read y < -0.5 out of 262144 - so this is a real swap and not a
// cleanup. It also reaches further than `probe_buried`: `trace` shares this
// function, so cascade rays change where they stop as well, and the frame
// moves as a result (see under `probe_visible`).
fn is_solid(texel: vec2<i32>) -> bool {
    let dims = textureDimensions(winding);
    if (texel.x < 0 || texel.y < 0 || texel.x >= i32(dims.x) || texel.y >= i32(dims.y)) {
        return false;
    }
    let t = vec2<u32>(texel);
    return textureLoad(ceiling_normal, t, 0).y < BURIED_MIN_Y
        || textureLoad(floor_normal, t, 0).y < BURIED_MIN_Y;
}

// Whether `t` is an open column a probe can be lifted from at all.
//
// Not solid, and with both stored surfaces, so there is a real interval to sit
// in. The interval has to be tall enough for the lift to land inside it. A
// column the bake found no ceiling in is open sky, and relocating a buried
// probe out of a wall into one of those is the worst possible answer: the probe
// would gather sky it can see from outside and pour it into the room it was
// just rescued from.
//
// No height is required here. The destination column places the probe wherever
// its own lift puts it, so a column that only opens above or below the probe's
// original height is still a valid - and much closer - escape than one that
// happens to straddle it.
fn column_open(t: vec2<i32>) -> bool {
    let dims = vec2<i32>(textureDimensions(heightmap));
    if (t.x < 0 || t.y < 0 || t.x >= dims.x || t.y >= dims.y) {
        return false;
    }
    if (is_solid(t)) {
        return false;
    }
    let u = vec2<u32>(t);
    let n_ceiling = textureLoad(ceiling_normal, u, 0).xyz;
    let n_floor = textureLoad(floor_normal, u, 0).xyz;
    if (dot(n_ceiling, n_ceiling) < 0.5 || dot(n_floor, n_floor) < 0.5) {
        return false;
    }
    let h = textureLoad(heightmap, u, 0).xy;
    return h.x - h.y >= 2.0 * PROBE_MIN_LIFT;
}

// Where a probe lifted from column `t` would end up: the column's own floor
// plus the same lift `probe_position` applies. Named so the search can measure
// how far a candidate actually moves the probe.
fn lifted_height(t: vec2<i32>) -> f32 {
    let h = textureLoad(heightmap, vec2<u32>(t), 0).xy;
    return h.y + clamp(0.5 * (h.x - h.y), PROBE_MIN_LIFT, PROBE_MAX_LIFT);
}

// The open column whose lift puts the probe closest to where it is now, within
// `ring` texels, in any direction.
//
// The rings grow in Chebyshev steps - they have to, to enumerate a square
// neighbourhood - but Chebyshev is only the enumeration. Every candidate in
// every ring is scored and the smallest wins, so the answer is the nearest
// escape rather than whichever one a square happened to reach first: stopping
// at the first ring with anything open picks by square, which on a diagonal
// settles for a texel up to sqrt(2) further off than one the same ring held.
//
// The score is in metres, not texels. A texel is 7.8 cm across in x and 6.2 cm
// in z, so texel-space distance silently over-weights x by a quarter, and it
// ignores height entirely - which is the axis a buried probe most often has to
// move along, since the destination column places the probe by its own lift.
// Scoring the full 3D move is what makes "any direction, closest" true rather
// than approximately true on a flat map.
fn escape_texel(origin: vec2<i32>, y: f32, ring: i32) -> vec2<i32> {
    if !is_solid(origin) {
        return origin;
    }
    let cell = cell_size();
    var best = origin;
    var best_d = -1.0;
    for (var r = 1; r <= ring; r = r + 1) {
        for (var i = -r; i <= r; i = i + 1) {
            var c: array<vec2<i32>, 4>;
            c[0] = origin + vec2<i32>(i, -r);
            c[1] = origin + vec2<i32>(i, r);
            c[2] = origin + vec2<i32>(-r, i);
            c[3] = origin + vec2<i32>(r, i);
            for (var k = 0; k < 4; k = k + 1) {
                if (column_open(c[k])) {
                    let dx = f32(c[k].x - origin.x) * cell.x;
                    let dz = f32(c[k].y - origin.y) * cell.y;
                    let dy = lifted_height(c[k]) - y;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if (best_d < 0.0 || d2 < best_d) {
                        best_d = d2;
                        best = c[k];
                    }
                }
            }
        }
    }
    return best;
}

// Whether the probe *as it ended up* is inside material, which after a nudge is
// not a question about the grid cell it was gathered for. Testing that cell
// instead would reject exactly the probes relocation just rescued, and accept
// a probe that never moved because its cell looked open while the position it
// is read at does not.
//
// The probe grid and the heightmap are not the same lattice: cascade 0 is one
// probe per metre, the heightmap one texel per 7.8 cm, so a grid index has to
// go through `probe_position`'s world-space round trip before it can index a
// heightmap texture. Handing a grid index straight to `is_solid` silently
// looks 12.8x/16x too far west and north, which for cascade 0 reaches only the
// map's far corner - 0 of Sponza's flagged texels, so the test rejected
// nothing anywhere in the scene while reading as if it were live. `p` is a
// world position, so that round trip has already happened by the time it gets
// here.
fn probe_buried(p: vec3<f32>) -> bool {
    return is_solid(texel_at(p.xz));
}

fn height_agreement(dy: f32, spacing: f32) -> f32 {
    let s = max(spacing * HEIGHT_TOLERANCE_SPACING, 1e-4);
    let t = abs(dy) / s;
    return 1.0 - smoothstep(HEIGHT_CUTOFF_SHOULDER, 1.0, t);
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
        // A solid column is material between its floor and ceiling, not the
        // open air a wall's [top, bottom] interval reads as. Block rays that
        // cross the band, at the face they entered through.
        if is_solid(texel) {
            hit.t = t;
            hit.ceiling = false;
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
