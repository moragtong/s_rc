// One-shot geometric bake for the cascades.
//
// The scene is static, so the ray marching that decides *where* each probe ray
// lands does not have to happen while the sun moves: this pass runs once, after
// the heightmap and the winding map are final, and records the hit for every
// ray. `cascades2_5d.wgsl` then only looks the radiance up at those stored
// hits, which is the part that changes when the lighting does.

#import s_rc::heightmap_trace::{
    uniforms, trace, Hit, probe_position, cell_direction, stored_segment, MISS,
}

// Per ray: the top cascade's own trace, or the four bilinear-parent segments
// the cascades below it trace instead. The two are mutually exclusive, so four
// slots cover both.
const HITS_PER_RAY: u32 = 4u;

@group(1) @binding(0) var<storage, read_write> hits: array<vec4<u32>>;

// texel.x, texel.y, ceiling (0 or 1), and the distance bitcast to u32. MISS
// keeps its own bits, so a slot that was never traced reads back as a miss.
fn store(base: u32, slot: u32, hit: Hit) {
    var e = vec4<u32>(0u, 0u, 0u, bitcast<u32>(MISS));
    if hit.t != MISS {
        e = vec4<u32>(
            u32(hit.texel.x),
            u32(hit.texel.y),
            select(0u, 1u, hit.ceiling),
            bitcast<u32>(hit.t),
        );
    }
    hits[base * HITS_PER_RAY + slot] = e;
}

fn store_miss(base: u32, slot: u32) {
    hits[base * HITS_PER_RAY + slot] = vec4<u32>(0u, 0u, 0u, bitcast<u32>(MISS));
}

// Per direction of the 2x2 parent block covering `ray`: the projected distance
// of that direction's nearest stored hit (min over its four slots), or -1 when
// it has none. y carries the direction's cell index, so `cascade()` can read
// the parent back in the directions that actually intersected something.
//
// The projection is the parallax mapping. The parent's hit is measured from
// the parent, so handing it straight to the child puts the endpoint somewhere
// off the child's own ray - which is what the old handoff did:
// `q + d * parent.interval.x` is a point in the *parent's* frame, and the
// segment then bent sideways to reach it. Projecting the hit point `s` onto
// the child's direction keeps the geometry and drops the offset:
//
//     t_child = dot(s - p, d) = t_parent + dot(q - p, d) .
//
// The nearest such distance is the endpoint the segment should be marched to,
// so march uses the same min `cascade()` does - the nearest candidate across
// all four directions and all four slots - and the segment then lies exactly
// along `d`, making `cascade()`'s `seg / len` equal `d`.
fn parent_dir_hits(
    cc: u32,
    qi: vec2<u32>,
    ray: vec2<u32>,
    p: vec3<f32>,
    d: vec3<f32>,
    t_start: f32,
) -> array<vec4<f32>, 4> {
    let info = uniforms.cascades[cc];
    // The top cascade's slots 1..3 are never written; only slot 0 is a trace.
    let slots = select(4u, 1u, cc + 1u >= uniforms.cascade_count);
    let ray_base = info.ray_base + (qi.y * info.size.x + qi.x) * info.octa_side * info.octa_side;
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
                let t = bitcast<f32>(hits[(ray_base + pr) * HITS_PER_RAY + k].w);
                if t == MISS {
                    continue;
                }
                let s = stored_segment(cc, qi, pr, k);
                let len = length(s.dir);
                if len <= 1e-5 {
                    continue;
                }
                let ts = dot(s.start + s.dir * (t / len) - p, d);
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

// Whether the child trace's segments end at the nearest projected parent hit.
// Off restores the old handoff. See `parent_dir_hits` above.
const PARALLAX_ENDPOINT: bool = true;

fn march(c: u32, id: u32) {
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
    let base = info.ray_base + id;

    if c + 1u >= uniforms.cascade_count {
        store(base, 0u, trace(p, d, info.interval.x, info.interval.y));
        return;
    }

    // The four segments the bilinear fix traces, in the same order and with
    // the same endpoints `cascade()` derives. They start at this ray's
    // interval start. Where they end is the parent's own hit point mapped
    // onto this ray by distance, so the near part of the ray is traced from
    // here instead of handed over where the parent's frame happens to say
    // its ray begins; when the parent has no hit to map, the old handoff
    // stands and the merge inherits as before.
    let parent = uniforms.cascades[c + 1u];
    let u = (p.xz - uniforms.hmap_min) / parent.spacing - 0.5;
    let cell = vec2<i32>(floor(u));
    let a = p + d * info.interval.x;
    let ray_cell = vec2<u32>(ray % side, ray / side);

    for (var k = 0u; k < HITS_PER_RAY; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let qi = vec2<u32>(
            clamp(cell + vec2<i32>(corner), vec2<i32>(0), vec2<i32>(parent.size) - 1),
        );
        let q = probe_position(c + 1u, qi);
        var end = q + d * parent.interval.x;
        let dirs = parent_dir_hits(c + 1u, qi, ray_cell, p, d, info.interval.x);
        var nearest = -1.0;
        for (var j = 0u; j < 4u; j++) {
            if dirs[j].x >= 0.0 && (nearest < 0.0 || dirs[j].x < nearest) {
                nearest = dirs[j].x;
            }
        }
        // A/B toggle: PARALLAX_ENDPOINT=false restores the pre-parallax
        // handoff, `end = q + d * parent.interval.x`, which the fallback above
        // always gives. Measured away from the interior camera runs.
        if PARALLAX_ENDPOINT && nearest >= 0.0 {
            end = p + d * nearest;
        }
        let seg = end - a;
        let len = length(seg);
        if len <= 1e-5 {
            // `cascade()` reads this as a miss and falls through to the parent
            // rays, exactly as it does when the segment is too short to trace.
            store_miss(base, k);
            continue;
        }
        store(base, k, trace(a, seg / len, 0.0, len));
    }
}

@compute @workgroup_size(64, 1, 1) fn march_0(@builtin(global_invocation_id) id: vec3<u32>) { march(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn march_1(@builtin(global_invocation_id) id: vec3<u32>) { march(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn march_2(@builtin(global_invocation_id) id: vec3<u32>) { march(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn march_3(@builtin(global_invocation_id) id: vec3<u32>) { march(3u, id.x); }
