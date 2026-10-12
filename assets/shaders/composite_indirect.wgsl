// Adds the indirect term back onto the lit HDR image. The shade pass renders
// the indirect into its own texture instead of blending it into the view
// target, so this pass is where that buffer gets consumed: it multiplies it by
// Bevy's SSAO AO before the additive blend puts it on the image.
//
// The AO multiply is gated by `info.ao_enabled`: when the SSAO component is
// absent the caller still binds a texture (the 1x1 white fallback) but flags
// it off, so `ao` reads 1.0 and the multiply is the identity. The flag is
// needed because `textureLoad` on the fallback returns 0 outside its single
// texel, which would otherwise zero the whole indirect term.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct CompositeUniforms {
    ao_enabled: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var indirect_tex: texture_2d<f32>;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> info: CompositeUniforms;

@fragment
fn composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let indirect = textureLoad(indirect_tex, pixel, 0).rgb;
    let ao = select(1.0, textureLoad(ao_tex, pixel, 0).r, info.ao_enabled != 0u);
    return vec4<f32>(indirect * ao, 0.0);
}