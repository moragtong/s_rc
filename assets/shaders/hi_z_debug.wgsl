// Fullscreen view of one Hi-Z mip.
//
// Gray: linear distance of the mip's (nearest) depth, white = close.
// Red: sky at full res, but the coarse cell holds geometry. That's the
// expected outward "fattening" of silhouettes, and the band should widen
// with each mip. If geometry shrinks away from the sky instead, the pyramid
// is keeping the farthest depth and the min/max is backwards.
// (Not "coarse closer than full" in general: on any sloped surface nearly
// every pixel has a nearer neighbor in its cell, so that floods the screen.)

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct HiZDebugUniform {
    mip: u32,
    near: f32,
    max_distance: f32,
}

@group(0) @binding(0) var hi_z: texture_2d<f32>;
@group(0) @binding(1) var<uniform> settings: HiZDebugUniform;

// Infinite reverse-Z perspective: ndc_z = near / view_distance.
fn linear_distance(ndc_z: f32) -> f32 {
    return settings.near / max(ndc_z, 1e-7);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let mip = min(settings.mip, textureNumLevels(hi_z) - 1u);
    let pixel = vec2<u32>(in.position.xy);
    let mip_size = textureDimensions(hi_z, mip);

    let full = textureLoad(hi_z, min(pixel, textureDimensions(hi_z, 0u) - 1u), 0).r;
    let coarse = textureLoad(hi_z, min(pixel >> vec2(mip), mip_size - 1u), mip).r;

    // Sky is 0 (far plane at infinity): show it black.
    let gray = select(
        0.0,
        1.0 - saturate(linear_distance(coarse) / settings.max_distance),
        coarse > 0.0,
    );

    if full == 0.0 && coarse > 0.0 {
        return vec4(mix(vec3(gray), vec3(1.0, 0.1, 0.1), 0.6), 1.0);
    }
    return vec4(vec3(gray), 1.0);
}
