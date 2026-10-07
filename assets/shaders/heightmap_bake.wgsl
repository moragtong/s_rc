// Reduces the two orthographic bake views into the heightmap: the up view
// gives each column's ceiling, the down view its floor.
//
// One dispatch for both views, because they write different channels of the
// same heightmap texture and a write-only storage texture cannot be read to
// preserve the other channel's value.
//
// Deliberately does not import heightmap_trace: that module declares its own
// @group(0) bindings, which would collide with these.

struct View {
    // world = world_from_view * (view_from_clip * vec4(ndc, depth, 1)).xyz / w
    view_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
}

struct Uniforms {
    up: View,
    down: View,
    hmap_res: vec2<u32>,
    // Ceiling height for open-sky columns. heightmap_trace treats any ray
    // going up past this as escaped, so it must stay above hmap_top.
    empty_ceiling: f32,
    empty_floor: f32,
    // The up camera is mirrored in x, so its texels map to the map reversed;
    // see the bake cameras in heightmap_bake.rs.
    up_flip_x: u32,
    down_flip_x: u32,
}

// Depth and deferred G-buffer attachments of each bake camera. Bevy clears
// reverse-Z depth to 0, so depth <= 0 means the far plane: that view saw
// nothing in that column.
@group(0) @binding(0) var up_depth: texture_depth_2d;
@group(0) @binding(1) var up_gbuffer: texture_2d<u32>;
@group(0) @binding(2) var down_depth: texture_depth_2d;
@group(0) @binding(3) var down_gbuffer: texture_2d<u32>;
@group(0) @binding(4) var<uniform> u: Uniforms;

@group(0) @binding(5) var heights: texture_storage_2d<rg32float, write>;
@group(0) @binding(6) var ceiling_normal: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var floor_normal: texture_storage_2d<rgba32float, write>;
@group(0) @binding(8) var ceiling_albedo: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var floor_albedo: texture_storage_2d<rgba32float, write>;

/// One surface as a bake view saw it. `hit` is false where the view's depth
/// stayed at the far plane, and everything else is then zero.
struct Surface {
    hit: bool,
    height: f32,
    normal: vec3<f32>,
    albedo: vec3<f32>,
}

/// Bevy's oct24 decode (bevy_pbr::render::utils::octahedral_decode), kept
/// local because heightmap_trace's @group(0) bindings would collide.
fn oct_decode(encoded: vec2<f32>) -> vec3<f32> {
    let f = encoded * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = saturate(-n.z);
    let w = select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0));
    n = vec3<f32>(n.xy + w, n.z);
    return normalize(n);
}

/// World position of the fragment at framebuffer `pixel` (in texels) at
/// `depth`, for this orthographic view.
fn world_at(view: View, depth: f32, pixel: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(
        pixel.x / f32(u.hmap_res.x) * 2.0 - 1.0,
        1.0 - pixel.y / f32(u.hmap_res.y) * 2.0,
    );
    let clip = vec4<f32>(ndc, depth, 1.0);
    let v = view.view_from_clip * clip;
    return (view.world_from_view * vec4<f32>(v.xyz / v.w, 1.0)).xyz;
}

fn surface(
    view: View,
    depth: f32,
    gbuffer: vec4<u32>,
    pixel: vec2<f32>,
    empty: f32,
) -> Surface {
    var s: Surface;
    s.hit = false;
    s.height = empty;
    s.normal = vec3<f32>(0.0);
    s.albedo = vec3<f32>(0.0);

    // G-buffer layout (bevy_pbr pbr_deferred_functions):
    //   r = base_color packed unorm4x8, g = emissive rgb9e5,
    //   b = reflectance/metallic/occlusion/clearcoat, a = oct24 normal+flags.
    let base_rough = unpack4x8unorm(gbuffer.r);
    // 2.2 rather than the exact sRGB curve, matching the deferred packer.
    let albedo = pow(base_rough.rgb, vec3<f32>(2.2));
    // a = two 12-bit oct components in [0,1], then 8 flag bits.
    let packed = vec2<f32>(
        f32(gbuffer.a & 0xFFFu),
        f32((gbuffer.a >> 12u) & 0xFFFu),
    ) / 4095.0;
    let normal = oct_decode(packed);

    // A pixel can clear depth without an opaque material having written the
    // G-buffer, leaving a zero normal; normalizing that is NaN and would
    // poison every cascade that reads this map.
    if depth <= 0.0 || dot(normal, normal) < 0.5 {
        return s;
    }

    s.hit = true;
    s.height = world_at(view, depth, pixel).y;
    s.normal = normal;
    s.albedo = albedo;
    return s;
}

/// Framebuffer texel (in pixels) that heightmap texel `texel` maps to in a
/// bake view.
fn framebuffer_texel(texel: vec2<u32>, flip_x: u32) -> vec2<f32> {
    let px = select(texel.x, u.hmap_res.x - 1u - texel.x, flip_x == 1u);
    return vec2<f32>(f32(px), f32(texel.y));
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.hmap_res) {
        return;
    }
    let up_px = framebuffer_texel(id.xy, u.up_flip_x);
    let down_px = framebuffer_texel(id.xy, u.down_flip_x);
    let up_i = vec2<i32>(up_px);
    let down_i = vec2<i32>(down_px);

    // Ceiling: lowest surface above the split, seen from below, so its normal
    // is flipped to point down out of the column. Floor: highest surface below
    // the split, seen from above.
    let ceiling = surface(
        u.up,
        textureLoad(up_depth, up_i, 0),
        textureLoad(up_gbuffer, up_i, 0),
        up_px,
        u.empty_ceiling,
    );
    let floor = surface(
        u.down,
        textureLoad(down_depth, down_i, 0),
        textureLoad(down_gbuffer, down_i, 0),
        down_px,
        u.empty_floor,
    );

    let t = vec2<i32>(id.xy);
    let zero = vec3<f32>(0.0);
    // rgba32float/rg32float storage writes take a vec4 in naga, even where
    // the format is narrower; the extra lanes are discarded.
    textureStore(heights, t, vec4<f32>(ceiling.height, floor.height, 0.0, 0.0));
    // A zero normal is how surface_light.wgsl and the cascades tell a texel
    // apart from one the bake found nothing in.
    textureStore(ceiling_normal, t, vec4<f32>(select(zero, ceiling.normal, ceiling.hit), 0.0));
    textureStore(floor_normal, t, vec4<f32>(select(zero, floor.normal, floor.hit), 0.0));
    textureStore(ceiling_albedo, t, vec4<f32>(select(zero, ceiling.albedo, ceiling.hit), 0.0));
    textureStore(floor_albedo, t, vec4<f32>(select(zero, floor.albedo, floor.hit), 0.0));
}