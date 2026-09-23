// Hi-Z pyramid for hierarchical screen-space ray marching.
//
// Each texel holds the NEAREST depth in its footprint. Bevy uses reverse-Z
// (1 = near plane, 0 = far plane), so "nearest" is max(). This is the
// opposite of Bevy's own occlusion-culling pyramid, which keeps the farthest
// depth (min) because culling must be conservative in the other direction.
// A march may only skip a cell if the ray is in front of everything in it,
// so the cell has to report its closest surface.
//
// FIRST_MIP copies the prepass depth buffer into mip 0 at full resolution,
// so the marcher can walk every level of one texture uniformly.

#ifdef FIRST_MIP
@group(0) @binding(0) var src: texture_depth_2d;
#else
@group(0) @binding(0) var src: texture_2d<f32>;
#endif
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

fn load_src(p: vec2<i32>) -> f32 {
#ifdef FIRST_MIP
    return textureLoad(src, p, 0);
#else
    return textureLoad(src, p, 0).r;
#endif
}

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(dst);
    if any(id.xy >= dst_size) {
        return;
    }

#ifdef FIRST_MIP
    let nearest = load_src(vec2<i32>(id.xy));
#else
    let src_size = vec2<i32>(textureDimensions(src));
    let base = vec2<i32>(id.xy) * 2;

    // With an odd source dimension, a plain 2x2 footprint never reads the
    // source's last row/column. The last destination texel absorbs it, so
    // thin geometry at the screen edge can't vanish from coarser mips.
    let last = vec2<i32>(dst_size) - 1;
    let extra = vec2<i32>(
        select(0, 1, (src_size.x & 1) == 1 && i32(id.x) == last.x),
        select(0, 1, (src_size.y & 1) == 1 && i32(id.y) == last.y),
    );

    var nearest = 0.0;
    for (var y = 0; y <= 1 + extra.y; y++) {
        for (var x = 0; x <= 1 + extra.x; x++) {
            nearest = max(nearest, load_src(min(base + vec2(x, y), src_size - 1)));
        }
    }
#endif

    textureStore(dst, id.xy, vec4(nearest, 0.0, 0.0, 0.0));
}
