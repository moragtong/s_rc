// Cross-fade between two relights: `sh_probes_blend = mix(prev, curr, alpha)`,
// one thread per SH word, run every frame so `shade` reads a single storage
// buffer instead of both relights.
//
// That matters because shade is by far the most expensive pass: for each of its
// 4 probes it samples the radiance 32 times through 9 SH coefficients, so a
// per-coefficient mix in the fragment shader doubled a 1.4-million-pixel pass's
// storage traffic. The whole probe set is ~2k probes x 10 words, so blending
// it here instead costs one ~20k-thread dispatch.
//
// Positions and parallax radii are identical in both sets - the geometry never
// moves, only the light on it - so they come through the mix unchanged. The
// pass shares group 0 with every other cascade pass; that group was previously
// unused here, but the low-pass below needs the per-cascade grid layout
// (`probe_base`, `size`) to find a probe's XZ neighbours.
//
// LOW-PASS. A probe's irradiance is the average over its cell, and neighbouring
// cells' probes are ~1 m apart at cascade 0, so the field carries a Nyquist
// ripple that shows as columns/blobs under direct lighting. After the temporal
// mix, each SH coefficient is smoothed by a 5-tap horizontal cross:
//
//     SH'(x,z) = 0.5*SH(x,z) + 0.125*[SH(x-1,z)+SH(x+1,z)+SH(x,z-1)+SH(x,z+1)]
//
// (partition of unity: 0.5 + 4*0.125 = 1.0). It only touches the 9 SH
// coefficients: the stride-10 word holds (position, radius), whose position
// differs between probes, so averaging it across neighbours would move the
// probe - that word is copied through the plain mix. Neighbours are clamped at
// the grid edge (replicate) and kept within the probe's own cascade and layer,
// so floor/ceiling and cascade boundaries never bleed. This runs every frame,
// but the inputs are the two raw relights (`prev`/`curr`), never the pass's own
// output, so there is no feedback and no cumulative over-blur.
//
// Each neighbour tap is also gated on vertical agreement with the center probe
// (`height_agreement_tol`, tolerance `LOWPASS_HEIGHT_TOL_SPACING` spacings) and
// the surviving weights renormalise, so an XZ step that the probes look across
// (a platform edge, a wall) does not bleed: the filter sharpens back towards
// the center probe instead of smoothing a discontinuity.
//
// The pre-filter body is kept commented at the bottom as the reference the
// filter is A/B'd against.

#import s_rc::heightmap_trace::{uniforms, height_agreement_tol, SH_STRIDE, PROBES_PER_CELL}

@group(1) @binding(0) var<storage, read> sh_probes_prev: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> sh_probes: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> sh_probes_blend: array<vec4<f32>>;

struct BlendUniforms {
    // 0 on the frame the probes were rewritten, 1 just before the next relight.
    alpha: f32,
    // Total vec4s of the probe set, so the dispatch knows where to stop.
    probe_words: u32,
    // 0 to skip the spatial low-pass entirely (plain cross-fade), 1 to filter.
    // The height gate is unconditional - there is no separate switch for it.
    lowpass: u32,
    // Uniform address space rounds the struct up to 16 bytes; the shader and
    // the CPU-side struct must agree on the binding's minimum size.
    pad0: u32,
}
@group(1) @binding(3) var<uniform> blend: BlendUniforms;

const FILTER_CENTER: f32 = 0.5;
const FILTER_NEIGHBOUR: f32 = 0.125;
// The neighbour gate's vertical tolerance, in cascade spacings. `height_agreement`
// itself defaults to 4 spacings (a 4 m step at cascade 0), which sees right
// through the room's surfaces; the low-pass uses this tighter value so a
// platform edge (~1.5 m) really cuts. Tune this to the discontinuity.
const LOWPASS_HEIGHT_TOL_SPACING: f32 = 1.5;

fn mixed_word(word: u32) -> vec4<f32> {
    return mix(sh_probes_prev[word], sh_probes[word], blend.alpha);
}

@compute @workgroup_size(64, 1, 1)
fn blend_sh(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= blend.probe_words {
        return;
    }
    let coeff = i % SH_STRIDE;
    let probe = i / SH_STRIDE;
    // The (position, radius) word is geometry, not light: positions differ
    // between probes, so it must not enter the spatial average.
    if coeff >= 9u || blend.lowpass == 0u {
        sh_probes_blend[i] = mixed_word(i);
        return;
    }

    // The cascade that owns this probe. `probe_base` is monotonic in cascade.
    var c = 0u;
    for (var k = 0u; k < 4u; k++) {
        let info_k = uniforms.cascades[k];
        if probe >= info_k.probe_base
            && probe < info_k.probe_base + info_k.size.x * info_k.size.y * PROBES_PER_CELL {
            c = k;
        }
    }
    let info = uniforms.cascades[c];
    let local = probe - info.probe_base;
    let layer = local % PROBES_PER_CELL;
    let cell = local / PROBES_PER_CELL;
    let px = cell % info.size.x;
    let py = cell / info.size.x;

    // The gate compares heights, so the center probe's position has to be
    // read too; it lives in the stride-10 word like every probe's does.
    let center_pos = mixed_word(probe * SH_STRIDE + 9u).xyz;

    var acc = mixed_word(i).xyz * FILTER_CENTER;
    // Renormalise: a neighbour that fails agreement is dropped and the
    // remaining weights must still sum to one, so the filter sharpens towards
    // the center probe rather than dimming.
    var wsum = FILTER_CENTER;
    for (var n = 0u; n < 4u; n++) {
        // Clamp at the edge (replicate) so the 5 weights always sum to 1.
        var qx = px;
        var qy = py;
        if n == 0u {
            if px > 0u { qx = px - 1u; }
        } else if n == 1u {
            if px + 1u < info.size.x { qx = px + 1u; }
        } else if n == 2u {
            if py > 0u { qy = py - 1u; }
        } else {
            if py + 1u < info.size.y { qy = py + 1u; }
        }
        let q_probe = info.probe_base + (qy * info.size.x + qx) * PROBES_PER_CELL + layer;
        // Unconditional gate: a neighbour that fails vertical agreement is
        // dropped and the remaining weights still sum to one, so the filter
        // sharpens towards the center probe rather than dimming.
        let q_pos = mixed_word(q_probe * SH_STRIDE + 9u).xyz;
        let wn = FILTER_NEIGHBOUR * height_agreement_tol(
            q_pos.y - center_pos.y,
            info.spacing * LOWPASS_HEIGHT_TOL_SPACING,
        );
        acc += mixed_word(q_probe * SH_STRIDE + coeff).xyz * wn;
        wsum += wn;
    }
    sh_probes_blend[i] = vec4<f32>(acc / max(wsum, 1e-5), 0.0);
}

// Reference (no low-pass), the A/B baseline. Swap it in for the body above:
//
// @compute @workgroup_size(64, 1, 1)
// fn blend_sh(@builtin(global_invocation_id) id: vec3<u32>) {
//     let i = id.x;
//     if i >= blend.probe_words {
//         return;
//     }
//     sh_probes_blend[i] = mix(sh_probes_prev[i], sh_probes[i], blend.alpha);
// }
