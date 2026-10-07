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

#import s_rc::heightmap_trace::{uniforms, probe_position, cell_direction, MISS}

@group(1) @binding(0) var radiance_min: texture_2d<f32>;
@group(1) @binding(1) var radiance_max: texture_2d<f32>;
// Per ray: rgb = radiance, a = distance from the probe to where it came
// from (MISS if nothing was hit).
@group(1) @binding(2) var<storage, read_write> rays: array<vec4<f32>>;
// Per ray: texel, ceiling flag and distance, as recorded by the one-shot
// march. Read-only — nothing here writes geometry back.
@group(1) @binding(3) var<storage, read> hits: array<vec4<u32>>;

const HITS_PER_RAY: u32 = 4u;

// A stored hit, left exactly as the march wrote it: (texel.x, texel.y,
// ceiling, t). naga will not build a `Hit` literal out of an imported type,
// so the encoding stays packed and is unpacked field by field instead.
fn hit_at(base: u32, slot: u32) -> vec4<u32> {
    return hits[base * HITS_PER_RAY + slot];
}

fn hit_t(e: vec4<u32>) -> f32 {
    return bitcast<f32>(e.w);
}

fn surface_radiance(e: vec4<u32>) -> vec3<f32> {
    let texel = vec2<i32>(e.xy);
    if e.z != 0u {
        return textureLoad(radiance_min, texel, 0).rgb;
    }
    return textureLoad(radiance_max, texel, 0).rgb;
}

fn cascade(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    let side = info.octa_side;
    let rays_per_probe = side * side;
    let probe_index = id / rays_per_probe;
    if probe_index >= info.size.x * info.size.y {
        return;
    }
    let ray = id % rays_per_probe;
    let p = probe_position(c, vec2<u32>(probe_index % info.size.x, probe_index / info.size.x));
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
    // ray's interval start to where that parent's ray of the same direction
    // starts, and take the parent's radiance only if nothing is in between.
    // Parents sit at their own column's height, so the segment also carries
    // the parallax between child and parent.
    let parent = uniforms.cascades[c + 1u];
    let parent_side = parent.octa_side;
    let u = (p.xz - uniforms.hmap_min) / parent.spacing - 0.5;
    let base = vec2<i32>(floor(u));
    let f = u - floor(u);
    let a = p + d * info.interval.x;
    // The 2x2 parent cells covering this ray's cell.
    let cell = vec2<u32>(ray % side, ray / side) * 2u;

    var radiance = vec3<f32>(0.0);
    var distance_sum = 0.0;
    var distance_weight = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let wxy = select(1.0 - f, f, corner == vec2<u32>(1u));
        let w = wxy.x * wxy.y;
        let qi = vec2<u32>(clamp(base + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(parent.size) - 1));
        let q = probe_position(c + 1u, qi);

        let seg = q + d * parent.interval.x - a;
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

        let parent_rays = parent.ray_base + (qi.y * parent.size.x + qi.x) * parent_side * parent_side;
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
