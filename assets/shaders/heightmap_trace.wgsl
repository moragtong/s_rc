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

/// Vertical tolerance of a probe footprint, in units of cascade spacing.
// Height tolerance between a probe's own height and the shaded pixel's, as a
// multiple of the cascade's spacing. Cascade-0 spacing is ~0.078 m, so 0.5 puts
// the cutoff at ~0.04 m. A probe is either at the pixel's height or it is
// somewhere else entirely: on another storey, or inside a wall whose column
// floor is the ground floor. There is no useful middle ground between the two,
// which is why this is a cutoff rather than a falloff.
const HEIGHT_TOLERANCE_SPACING: f32 = 2.0;

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
    let t0 = texel_at(xz);

    // Nudge out of the wall the bake recorded this column inside, before the
    // lift: the lift measures the column's floor and ceiling, so relocating
    // first is what makes the probe end up in the air it was moved into rather
    // than at the same height inside the material it left.
    //
    // How far it may look depends on the cascade. Each one gathers from its own
    // `interval.x` outward - 0, 1, 2, 4 m - so a coarse cascade's rays start
    // beyond a wall that cascade-0's still run into, and it can afford to send
    // the probe further to clear it. Half a spacing caps the reach, because
    // past that the probe has left the bilinear footprint that reads it and is
    // dropped for being adrift rather than for being buried.
    let reach = max(RELOCATE_MIN_RADIUS, min(info.interval.x, 0.5 * s));
    let ring = max(i32(ceil(reach / min(cell_size().x, cell_size().y))), 1);
    let t = escape_texel(t0, lifted_height(t0), ring);

    // A probe that moved is read at its new position, so it goes there. One
    // that did not keeps its grid position to the millimetre: snapping every
    // probe to a texel centre would shift the whole field by up to half a
    // texel and change the image for no reason.
    var xz_out = xz;
    if any(t != t0) {
        xz_out = uniforms.hmap_min + (vec2<f32>(t) + 0.5) * cell_size();
    }

    let h = textureLoad(heightmap, vec2<u32>(t), 0).xy;
    let lift = clamp(0.5 * (h.x - h.y), PROBE_MIN_LIFT, PROBE_MAX_LIFT);
    let nominal = h.y + lift;
    let p = vec3<f32>(xz_out.x, nominal, xz_out.y);

    let n_ceiling = textureLoad(ceiling_normal, vec2<u32>(t), 0).xyz;
    let n_floor = textureLoad(floor_normal, vec2<u32>(t), 0).xyz;
    // A zero normal is how the bake marks a texel it found no surface in, and
    // those columns keep their sentinel height. There is no surface there to
    // push out of, so relocation has to leave them alone: with an empty floor
    // at EMPTY_FLOOR the lift clamp would otherwise fire on every one of them.
    let has_ceiling = dot(n_ceiling, n_ceiling) > 0.5;
    let has_floor = dot(n_floor, n_floor) > 0.5;
    if !has_floor || !has_ceiling {
        return p;
    }

    // Solid is everything outside (h.y, h.x); the interval itself is the open
    // column. PROBE_MIN_LIFT can push the nominal position straight through a
    // thin ceiling - a column 5 cm tall still gets lifted 10 cm - which parks
    // the probe inside the surface meant to bound it, and a probe inside
    // solid gathers the light of whatever it is buried in.
    let d_floor = nominal - h.y;
    let d_ceiling = h.x - nominal;
    if d_floor >= 0.0 && d_ceiling >= 0.0 {
        return p;
    }

    // Buried: leave along the normal of the nearer of the two surfaces, which
    // is the one whose solid the probe is inside. Both stored normals already
    // point out of their surface into the open column - the ceiling's is
    // negated at bake time so that it points down - so the same expression
    // works for either branch.
    let n = select(n_floor, n_ceiling, d_ceiling < d_floor);
    let overshoot = max(-d_ceiling, -d_floor) + PROBE_SURFACE_EPS;
    // |n.y| is the fraction of that distance that goes into Y, so the push
    // lands exactly PROBE_SURFACE_EPS clear of the surface whatever its tilt.
    return p + n * (overshoot / max(abs(n.y), PROBE_NORMAL_MIN_Y));
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
// A floor and a ceiling cannot say "this column is inside a wall", which is what
// let probes form inside masonry and collect a bilinear share from both
// storeys. The normals can, for free, because the bake cameras rasterise
// backfaces: a column that starts in open air records the surface its ray lands
// on, while a column that starts inside a wall records the surface the ray
// *leaves* through - the wall's underside going down, its top going up. Both of
// those face away from the split, and the bake negates the ceiling's normal so
// that both channels land on the same sign: a negative y.
//
// Either channel is enough, and either alone would throw most of it away. The
// two cameras look at opposite sides of the split, so whichever surface a buried
// column happens to be made of is only visible to one of them: on Sponza the
// floor channel catches the perimeter band and the ceiling channel catches the
// colonnade, and requiring both would keep 468 of the 2300 flagged texels.
//
// What neither channel can see is a vertical surface, because the bake's rays
// run only up and down and so never cross one: both sides of a wall record open
// air, no normal leans away, and 115 probes sit inside the drawn sphere of a
// wall with every one of their columns called open. The winding term is the
// missing half - a horizontal cut through the same geometry, which a wall
// crosses head-on. It is an addition, not a replacement: on Sponza it reproduces
// 98.5% of what the normals already catch and adds columns they cannot reach.
fn is_solid(texel: vec2<i32>) -> bool {
    let dims = textureDimensions(ceiling_normal);
    if (texel.x < 0 || texel.y < 0 || texel.x >= i32(dims.x) || texel.y >= i32(dims.y)) {
        return false;
    }
    // A zero normal is how the bake marks a column it found nothing in, which
    // is empty rather than buried, and compares greater than the threshold
    // anyway - so the empty columns need no case of their own.
    let t = vec2<u32>(texel);
    return textureLoad(ceiling_normal, t, 0).y < BURIED_MIN_Y
        || textureLoad(floor_normal, t, 0).y < BURIED_MIN_Y
        || abs(textureLoad(winding, t, 0).x) > WINDING_SOLID;
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
