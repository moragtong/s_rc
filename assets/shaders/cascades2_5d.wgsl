// Merges one cascade, one thread per probe ray, top cascade first. Each ray
// stores the radiance arriving along it from its interval start outward (its
// own interval merged with every cascade above), and how far away that
// radiance came from.
//
// Where each ray lands was decided once, by `march_hits.wgsl`: the scene is
// static, so the marching itself is never repeated here. This pass only looks
// the current surface radiance up at the stored hits and folds it into the
// cascades, which is what makes relighting cheap enough to run on its own
// schedule.

#import s_rc::heightmap_trace::{
    uniforms, probe_position, cell_direction, stored_segment, parent_layer, MISS, PROBES_PER_CELL, PI,
}

@group(1) @binding(0) var normal_min: texture_2d<f32>;
@group(1) @binding(1) var normal_max: texture_2d<f32>;
@group(1) @binding(2) var albedo_min: texture_2d<f32>;
@group(1) @binding(3) var albedo_max: texture_2d<f32>;
// Per ray: rgb = radiance, a = distance from the probe to where it came
// from (MISS if nothing was hit).
@group(1) @binding(4) var<storage, read_write> rays: array<vec4<f32>>;
// Per ray: texel, ceiling flag, cached horizon angles, and distance, as recorded by march_hits.
@group(1) @binding(5) var<storage, read> hits: array<vec4<u32>>;

const HITS_PER_RAY: u32 = 4u;

fn hit_at(base: u32, slot: u32) -> vec4<u32> {
    return hits[base * HITS_PER_RAY + slot];
}

fn hit_t(e: vec4<u32>) -> f32 {
    return bitcast<f32>(e.w);
}

// Evaluates surface radiance on-the-fly from albedo, normal, and the cached
// Aaltonen / Quilez directional horizon profile stored in the hit:
fn surface_radiance(e: vec4<u32>) -> vec3<f32> {
    let texel = vec2<i32>(e.xy);
    let is_ceiling = (e.z & 1u) != 0u;

    let n = select(
        textureLoad(normal_max, texel, 0).xyz,
        textureLoad(normal_min, texel, 0).xyz,
        is_ceiling,
    );
    if dot(n, n) <= 0.01 {
        return vec3<f32>(0.0);
    }
    let n_unit = normalize(n);
    let cos_l = dot(n_unit, uniforms.sun_toward);
    if cos_l <= 0.0 {
        return vec3<f32>(0.0);
    }

    let albedo = select(
        textureLoad(albedo_max, texel, 0).rgb,
        textureLoad(albedo_min, texel, 0).rgb,
        is_ceiling,
    );

    // Unpack cached directional horizons (+X, -X, +Z, -Z):
    let packed_h = e.z >> 1u;
    let h_px = f32(packed_h & 255u) / 255.0 * (PI * 0.5);
    let h_nx = f32((packed_h >> 8u) & 255u) / 255.0 * (PI * 0.5);
    let h_pz = f32((packed_h >> 16u) & 255u) / 255.0 * (PI * 0.5);
    let h_nz = f32((packed_h >> 24u) & 255u) / 255.0 * (PI * 0.5);

    // Interpolate horizon elevation along the sun azimuth:
    let sun_xz = uniforms.sun_toward.xz;
    let len_xz = length(sun_xz);
    var horizon_elev = 0.0;
    if len_xz > 1e-4 {
        let dir_xz = sun_xz / len_xz;
        let wx = abs(dir_xz.x);
        let wz = abs(dir_xz.y);
        let hx = select(h_nx, h_px, dir_xz.x >= 0.0);
        let hz = select(h_nz, h_pz, dir_xz.y >= 0.0);
        horizon_elev = (wx * hx + wz * hz) / max(wx + wz, 1e-4);
    }

    // Sun elevation angle above horizontal plane:
    let sun_elev = asin(clamp(uniforms.sun_toward.y, -1.0, 1.0));

    // Aaltonen soft shadow penumbra factor from angular horizon clearance:
    // delta_angle in radians; theta_sun = 0.27 deg (~0.0047 rad)
    let theta_sun = max(1.0 / uniforms.sun_k, 0.002);
    let delta = sun_elev - horizon_elev;
    let res = clamp((delta + theta_sun) / (2.0 * theta_sun), 0.0, 1.0);
    let vis = res * res * (3.0 - 2.0 * res);

    return albedo / PI * uniforms.sun_irradiance * cos_l * vis;
}

// Distance along this ray at which the parent probe's nearest stored surface
// appears, with the parent direction cell it came from; x is -1 when the
// parent has none in this direction. **Must give the same answer
// `march_hits.wgsl` gave**, or the stored hit's `t` would be measured along a
// different segment than the one derived here; only the reads differ (its
// group binds `hits` at binding 0, this one at binding 3).
//
// The mapping is the parallax correction: the parent's hit is measured from
// the parent, so the endpoint `q + d * parent.interval.x` - what this used to
// use - is a point in the *parent's* frame and the segment bent sideways to
// reach it. Projecting the parent's hit point `s` onto the child's direction
// keeps the geometry and drops the offset,
//
//     t_child = dot(s - p, d) = t_parent + dot(q - p, d),
//
// so the segment lies along `d`, `seg / len` below is `d`, and the distance
// stored for the child is its own distance along its own ray.
//
// Candidates are the parent's four direction cells covering this ray (twice
// the angular resolution, so this cell is a 2x2 block of the parent's) times
// that ray's four stored segments - the parent probe's own hit points, not the
// ones it inherited from coarser cascades. Nearest wins: the segment is a
// first-hit search and only has to reach the closest surface the parent
// already knows about. `y` is the direction cell that surface was seen in,
// which is the direction the parent's radiance for it is stored under - so the
// merge below reads *that* ray rather than averaging the whole block, which
// is what it does when there is no hit to correct against.
fn parent_dir_hits(
    cc: u32,
    qi: vec2<u32>,
    ray: vec2<u32>,
    p: vec3<f32>,
    d: vec3<f32>,
    t_start: f32,
    layer: u32,
) -> array<vec4<f32>, 4> {
    let info = uniforms.cascades[cc];
    // The top cascade's slots 1..3 are never written; only slot 0 is a trace.
    let slots = select(4u, 1u, cc + 1u >= uniforms.cascade_count);
    let ray_base = info.ray_base
        + ((qi.y * info.size.x + qi.x) * PROBES_PER_CELL + layer) * info.octa_side * info.octa_side;
    var out = array<vec4<f32>, 4>(
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
    );
    for (var by = 0u; by < 2u; by++) {
        for (var bx = 0u; bx < 2u; bx++) {
            let pc = ray * 2u + vec2<u32>(bx, by);
            let pr = pc.y * info.octa_side + pc.x;
            var nearest = -1.0;
            for (var k = 0u; k < slots; k++) {
                let e = hit_at(ray_base + pr, k);
                if hit_t(e) == MISS {
                    continue;
                }
                let s = stored_segment(cc, qi, pr, k, layer);
                let len = length(s.dir);
                if len <= 1e-5 {
                    continue;
                }
                let ts = dot(s.start + s.dir * (hit_t(e) / len) - p, d);
                if ts > t_start && (nearest < 0.0 || ts < nearest) {
                    nearest = ts;
                }
            }
            if nearest >= 0.0 {
                out[by * 2u + bx] = vec4<f32>(nearest, f32(pr), 0.0, 0.0);
            }
        }
    }
    return out;
}

fn cascade(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    let side = info.octa_side;
    let rays_per_probe = side * side;
    let probe_index = id / rays_per_probe;
    if probe_index >= info.size.x * info.size.y * PROBES_PER_CELL {
        return;
    }
    let layer = probe_index % PROBES_PER_CELL;
    let cell_id = probe_index / PROBES_PER_CELL;
    let qi_child = vec2<u32>(cell_id % info.size.x, cell_id / info.size.x);
    let ray = id % rays_per_probe;
    let p = probe_position(c, qi_child, layer);
    let d = cell_direction(ray, side);

    if c + 1u >= uniforms.cascade_count {
        // A ray that hit nothing reached the open sky: store its radiance in
        // the MISS slot so the parent merges treat it like any other surface.
        var out = vec4<f32>(uniforms.sky_color.rgb, MISS);
        let e = hit_at(info.ray_base + id, 0u);
        let t = hit_t(e);
        if t != MISS {
            out = vec4<f32>(surface_radiance(e), t);
        }
        rays[info.ray_base + id] = out;
        return;
    }

    // Bilinear fix: toward each of the 4 bilinear parents, trace from this
    // ray's interval start to where the parent's own hit appears along this
    // ray, and take the parent's radiance only if nothing is in between.
    // The endpoint is distance-mapped rather than raw - see
    // `parent_dir_hits` - so the segment runs along `d` and the offset
    // between the two probes is carried by the endpoint's distance instead of
    // by a sideways bend.
    let parent = uniforms.cascades[c + 1u];
    let parent_side = parent.octa_side;
    let u = (p.xz - uniforms.hmap_min) / parent.spacing - 0.5;
    let base = vec2<i32>(floor(u));
    let f = u - floor(u);
    let a = p + d * info.interval.x;
    let ray_cell = vec2<u32>(ray % side, ray / side);
    // The 2x2 parent cells covering this ray's cell.
    let cell = ray_cell * 2u;

    var radiance = vec3<f32>(0.0);
    var distance_sum = 0.0;
    var distance_weight = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let wxy = select(1.0 - f, f, corner == vec2<u32>(1u));
        let w = wxy.x * wxy.y;
        let qi = vec2<u32>(clamp(base + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(parent.size) - 1));
        // The parent side of the merge follows `parent_layer`: the parent probe
        // position, its stored segment geometry and its ray block all read the
        // same layer - the child's own by default, the other layer's when
        // `cross_layer` is on.
        let q = probe_position(c + 1u, qi, parent_layer(layer));

        // Same endpoint the march traced: the nearest of the parent's hits
        // projected onto this ray, falling back to the parent's ray start when
        // it has none in this direction - which is the case the march stored a
        // miss for.
        var end = q + d * parent.interval.x;
        // Endpoint parallax is off (see `parallax_endpoint`): the old handoff
        // above stands. Turning the uniform on again computes the parent's hit
        // candidates and ends the segment where they project onto this ray.
        var n_dirs = 0u;
        var nearest = -1.0;
        if uniforms.parallax_endpoint != 0u {
            let dirs = parent_dir_hits(c + 1u, qi, ray_cell, p, d, info.interval.x, parent_layer(layer));
            for (var j = 0u; j < 4u; j++) {
                if dirs[j].x >= 0.0 {
                    n_dirs += 1u;
                    if nearest < 0.0 || dirs[j].x < nearest {
                        nearest = dirs[j].x;
                    }
                }
            }
            if n_dirs > 0u {
                end = p + d * nearest;
            }
        }
        let seg = end - a;
        let len = length(seg);
        // Too short to have been traced: the march stored a miss, so this
        // falls through to the parent rays below, as it always did.
        let e = hit_at(info.ray_base + id, k);
        let t = hit_t(e);
        if t != MISS {
            let dir = seg / len;
            radiance += w * surface_radiance(e);
            distance_sum += w * distance(p, a + dir * t);
            distance_weight += w;
            continue;
        }

        let parent_rays = parent.ray_base
            + ((qi.y * parent.size.x + qi.x) * PROBES_PER_CELL + parent_layer(layer))
                * parent_side
                * parent_side;
        // The inheritance is the full 2x2 parent block, always: the child's
        // direction sits centrally in that block, and pruning the directions
        // that ended in sky or whose projected point is short of the child's
        // interval throws out answers the parent already owns - the sky
        // especially, whose `MISS` slot carries `sky_color` for exactly this.
        // Radiance Cascades gathers from all four parent rays unconditionally.
        var l = vec3<f32>(0.0);
        var parent_distance = 0.0;
        var parent_hits = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let pc = cell + vec2<u32>(j & 1u, j >> 1u);
            let pr = pc.y * parent_side + pc.x;
            let r = rays[parent_rays + pr];
            l += 0.25 * r.rgb;
            if r.a != MISS {
                parent_distance += distance(p, q + cell_direction(pr, parent_side) * r.a);
                parent_hits += 1.0;
            }
        }
        radiance += w * l;
        if parent_hits > 0.0 {
            distance_sum += w * parent_distance / parent_hits;
            distance_weight += w;
        }
    }
    let dist = select(MISS, distance_sum / distance_weight, distance_weight > 0.0);
    rays[info.ray_base + id] = vec4<f32>(radiance, dist);
}

@compute @workgroup_size(64, 1, 1) fn cascade_0(@builtin(global_invocation_id) id: vec3<u32>) { cascade(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn cascade_1(@builtin(global_invocation_id) id: vec3<u32>) { cascade(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn cascade_2(@builtin(global_invocation_id) id: vec3<u32>) { cascade(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn cascade_3(@builtin(global_invocation_id) id: vec3<u32>) { cascade(3u, id.x); }
