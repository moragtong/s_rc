// Cross-fade between two relights: `sh_probes_blend = mix(prev, curr, alpha)`,
// one thread per SH word, run every frame so `shade` reads a single storage
// buffer instead of both relights.
//
// That matters because shade is by far the most expensive pass: for each of its
// 4 probes it samples the radiance 32 times through 9 SH coefficients, so a
// per-coefficient mix in the fragment shader doubled a 1.4-million-pixel pass's
// storage traffic. The whole probe set is 1839 probes x 10 words, so blending
// it here instead costs one 18k-thread dispatch.
//
// Positions and parallax radii are identical in both sets - the geometry never
// moves, only the light on it - so they come through the mix unchanged. The
// pass shares group 0 with every other cascade pass and simply reads nothing
// from it; all it needs is the alpha below.

@group(1) @binding(0) var<storage, read> sh_probes_prev: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> sh_probes: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> sh_probes_blend: array<vec4<f32>>;

struct BlendUniforms {
    // 0 on the frame the probes were rewritten, 1 just before the next relight.
    alpha: f32,
    // Total vec4s of the probe set, so the dispatch knows where to stop.
    probe_words: u32,
    pad0: u32,
    pad1: u32,
}
@group(1) @binding(3) var<uniform> blend: BlendUniforms;

@compute @workgroup_size(64, 1, 1)
fn blend_sh(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= blend.probe_words {
        return;
    }
    sh_probes_blend[i] = mix(sh_probes_prev[i], sh_probes[i], blend.alpha);
}
