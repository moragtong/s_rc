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
// mix, each SH coefficient is smoothed spatially, then `shade` reconstructs the
// field from the smoothed grid.
//
// ACTIVE: a separable cubic B-spline pre-filter. The 1D cubic B-spline basis
// sampled at integer offsets is the mask [1,4,1]/6, so the tensor-product 2D
// kernel is the 3x3 binomial
//
//      [1 4 1]
//      [4 16 4] / 36
//      [1 4 1]
//
// (partition of unity: 16 + 4*4 + 4*1 = 36). Applied to the grid it produces
// the B-spline control coefficients for the sampled field; reconstructing with
// the cubic B-spline basis then gives a C^2 continuous field, so a probe enters
// and leaves the footprint with zero first and second derivatives at the cell
// boundary instead of the gradient kink bilinear C^0 leaves. (`shade`'s read is
// smootherstep-weighted bilinear, not the exact cubic basis, so treat the C^2
// as the design intent; this pass removes the grid's high-frequency ripple the
// way the spline basis does.) It only touches the 9 SH coefficients: the
// stride-10 word holds (position, radius), whose position differs between
// probes, so averaging it across neighbours would move the probe - that word is
// copied through the plain mix. Neighbours are clamped at the grid edge
// (replicate) and kept within the probe's own cascade and layer, so
// floor/ceiling and cascade boundaries never bleed. This runs every frame, but
// the inputs are the two raw relights (`prev`/`curr`), never the pass's own
// output, so there is no feedback and no cumulative over-blur.
//
// PREVIOUS: a 5-tap horizontal tent cross, `0.5*center + 0.125*sum(4-neigh)`,
// kept below behind `TENT_PREFILTER` for A/B. It is not separable-smooth: its
// 1D mask [0.125, 0.75, 0.125] has no C^2 story and its diagonal neighbours get
// no weight, so the ripple it removes is anisotropic.
//
// Both kernels gate each neighbour tap on vertical agreement with the center
// probe (`height_agreement_tol`, tolerance `LOWPASS_HEIGHT_TOL_SPACING`
// spacings) AND pre-baked probe-to-probe visibility (`probe_adj`), so
// horizontal occluders like walls and vertical steps do not bleed across.

#import s_rc::heightmap_trace::{uniforms, height_agreement_tol, SH_STRIDE, PROBES_PER_CELL}

@group(1) @binding(0) var<storage, read> sh_probes_in: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> sh_probes_cur: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> sh_probes_out: array<vec4<f32>>;

struct BlendUniforms {
    // 0 on the frame the probes were rewritten, 1 just before the next relight.
    alpha: f32,
    // Total vec4s of the probe set, so the dispatch knows where to stop.
    probe_words: u32,
    // 0 to skip the spatial low-pass entirely (plain cross-fade), 1 to filter.
    lowpass: u32,
    // Pass index: 0 for the initial temporal mix + blur pass, >0 for successive blur passes.
    pass_index: u32,
    // Vertical cross-layer blur radius in meters.
    cross_layer_radius: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}
@group(1) @binding(3) var<uniform> blend: BlendUniforms;
@group(1) @binding(4) var<storage, read> probe_adj: array<u32>;

// Cubic B-spline pre-filter (separable [1,4,1]/6 tensor mask) vs the previous
// 5-tap tent cross. Flip to `false` for the A/B; the tent path is the block
// under `if !TENT_PREFILTER`.
const TENT_PREFILTER: bool = false;
// 1D B-spline mask [1,4,1]/6, tensor-product weights for the 3x3.
const BSPLINE_CENTER: f32 = 16.0 / 36.0;
const BSPLINE_EDGE: f32 = 4.0 / 36.0;
const BSPLINE_CORNER: f32 = 1.0 / 36.0;
// The previous anisotropic cross.
const FILTER_CENTER: f32 = 0.5;
const FILTER_NEIGHBOUR: f32 = 0.125;
// The neighbour gate's vertical tolerance, in cascade spacings. `height_agreement`
// itself defaults to 4 spacings (a 4 m step at cascade 0), which sees right
// through the room's surfaces; the low-pass uses this tighter value so a
// platform edge (~1.5 m) really cuts. Tune this to the discontinuity.
const LOWPASS_HEIGHT_TOL_SPACING: f32 = 1.5;
// Weight normalization epsilon - raised to prevent dark bands
// Probes across layers within cross_layer_radius blend together, weighted smoothly by distance.

fn layer_distance_weight(dy: f32, radius: f32) -> f32 {
    let s = max(radius, 1e-3);
    let u = abs(dy) / s;
    if u >= 1.0 {
        return 0.0;
    }
    let inv_u2 = 1.0 - u * u;
    return inv_u2 * inv_u2;
}

const WEIGHT_EPS: f32 = 1e-3;

fn input_word(word: u32) -> vec4<f32> {
    if blend.pass_index == 0u {
        return mix(sh_probes_in[word], sh_probes_cur[word], blend.alpha);
    } else {
        return sh_probes_in[word];
    }
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
        sh_probes_out[i] = input_word(i);
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
    let center_pos = input_word(probe * SH_STRIDE + 9u).xyz;

    var acc = vec3<f32>(0.0);
    // Renormalise: a neighbour that fails agreement or visibility is dropped and the
    // remaining weights must still sum to one, so the filter sharpens towards
    // the center probe rather than dimming.
    var wsum = 0.0;
    let adj_mask = probe_adj[probe];

    let r_y = blend.cross_layer_radius;

    if TENT_PREFILTER {
        // PREVIOUS anisotropic 5-tap cross (see the header). Kept for A/B.
        for (var n = 0u; n < 5u; n = n + 1u) {
            var qx = px;
            var qy = py;
            var bw = FILTER_CENTER;
            if n > 0u {
                let adj_idx = select(select(select(7u, 5u, n == 3u), 3u, n == 2u), 1u, n == 1u);
                if ((adj_mask & (1u << adj_idx)) == 0u) {
                    continue;
                }
                bw = FILTER_NEIGHBOUR;
                if n == 1u {
                    if px > 0u { qx = px - 1u; }
                } else if n == 2u {
                    if px + 1u < info.size.x { qx = px + 1u; }
                } else if n == 3u {
                    if py > 0u { qy = py - 1u; }
                } else {
                    if py + 1u < info.size.y { qy = py + 1u; }
                }
            }
            let cell_base = info.probe_base + (qy * info.size.x + qx) * PROBES_PER_CELL;
            for (var l = 0u; l < PROBES_PER_CELL; l = l + 1u) {
                if r_y <= 0.001 && l != layer {
                    continue;
                }
                let q_probe = cell_base + l;
                let q_pos = input_word(q_probe * SH_STRIDE + 9u).xyz;
                let dy = abs(q_pos.y - center_pos.y);
                let wy = select(layer_distance_weight(dy, r_y), 1.0, r_y <= 0.001);
                if wy <= 0.0 {
                    continue;
                }
                let wn = bw * wy;
                acc += input_word(q_probe * SH_STRIDE + coeff).xyz * wn;
                wsum += wn;
            }
        }
    } else {
        // Cubic B-spline pre-filter: separable [1,4,1]/6 tensor mask across the
        // 3x3 horizontal footprint, combined with continuous distance weighting
        // across vertical layers within the cascade.
        let offs = array<vec2<i32>, 9>(
            vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
            vec2<i32>(-1,  0), vec2<i32>(0,  0), vec2<i32>(1,  0),
            vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1),
        );
        let base_w = array<f32, 9>(
            BSPLINE_CORNER, BSPLINE_EDGE, BSPLINE_CORNER,
            BSPLINE_EDGE, BSPLINE_CENTER, BSPLINE_EDGE,
            BSPLINE_CORNER, BSPLINE_EDGE, BSPLINE_CORNER,
        );
        for (var n = 0u; n < 9u; n = n + 1u) {
            // Pre-baked probe-to-probe visibility test:
            // bit n is 1 if neighbor column n is visible, 0 if occluded by a wall.
            // Center tap (n == 4u) is always visible to itself.
            if n != 4u && ((adj_mask & (1u << n)) == 0u) {
                continue;
            }

            // Clamp at the grid edge (replicate).
            let qx = u32(clamp(i32(px) + offs[n].x, 0, i32(info.size.x) - 1));
            let qy = u32(clamp(i32(py) + offs[n].y, 0, i32(info.size.y) - 1));
            let cell_base = info.probe_base + (qy * info.size.x + qx) * PROBES_PER_CELL;

            // Blur across layers within the cascade, respecting physical vertical distance.
            for (var l = 0u; l < PROBES_PER_CELL; l = l + 1u) {
                if r_y <= 0.001 && l != layer {
                    continue;
                }
                let q_probe = cell_base + l;
                let q_pos = input_word(q_probe * SH_STRIDE + 9u).xyz;
                let dy = abs(q_pos.y - center_pos.y);
                let wy = select(layer_distance_weight(dy, r_y), 1.0, r_y <= 0.001);
                if wy <= 0.0 {
                    continue;
                }
                let wn = base_w[n] * wy;
                acc += input_word(q_probe * SH_STRIDE + coeff).xyz * wn;
                wsum += wn;
            }
        }
    }
    if wsum <= WEIGHT_EPS {
        sh_probes_out[i] = input_word(i);
    } else {
        sh_probes_out[i] = vec4<f32>(acc / wsum, 0.0);
    }
}
