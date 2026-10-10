// Instanced probe-sphere debug overlay, replacing the old ray-traced
// `probe_spheres` in the shade pass (which read the probe grid along each
// view ray and came out effectively invisible even at 0.4x spacing).
//
// This pass rasterizes real geometry: one unit icosphere, instanced once per
// probe-with-layer across every cascade in a single draw call. The per-instance
// world position comes straight from `probe_positions` - the same buffer the
// trace/gather/shade passes read - so the spheres sit at the baked positions
// exactly: floor and ceiling layers at their real surface heights, the
// `escape_texel` sideways nudges included. It writes to the window camera's
// framebuffer and depth-tests against the scene's prepass depth, so geometry
// occludes the spheres.
//
// Each sphere is shaded by the SH probe it stands for: the fragment reads that
// probe's 9 coefficients out of `sh_probes` (the same cross-faded buffer the
// shade pass samples) and evaluates the clamped-cosine irradiance along the
// sphere's own world normal. So a sphere shows the light its probe captured -
// floor-layer probes facing up toward the ceiling, ceiling-layer probes facing
// down - instead of a flat per-cascade tint. The blend is opaque (the spheres
// are the subject, not an overlay on the lit scene).
//
// `probe_positions` is two vec4s per probe-with-layer (even = position,
// odd = normal). The concatenation order matches the cascades' `sh_base`
// ranges, so a probe's cascade - and its radius, baked into the cascade's
// `params.z` on the CPU side as spacing * scale - is read off those ranges.
// The SH buffer uses the same probe-with-layer index at stride `SH_STRIDE`.

#import s_rc::heightmap_trace::{SH_STRIDE, sh_basis, PI}

const CASCADES: u32 = 4u;
// Scale on the probe's irradiance before the pre-tonemap write. Irradiance is
// radiance convolved with the clamped cosine, so it runs higher than a scene
// radiance; 1.0 puts the lit spheres in the same range as the lit scene and
// lets tonemapping compress the bright -Y (sky) side.
const PROBE_BRIGHTNESS: f32 = 1.0;

struct ProbeOverlayUniforms {
    clip_from_world: mat4x4<f32>,
    // One vec4 per cascade, in cascade order:
    //   x = first probe-with-layer index, y = probe count, z = sphere radius.
    params: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> uniforms: ProbeOverlayUniforms;
@group(0) @binding(1) var<storage, read> probe_positions: array<vec4<f32>>;
// Cross-faded SH probes, `SH_STRIDE` vec4 words per probe-with-layer: words
// 0..8 are the coefficients (xyz), word 9 is (position.xyz, radius).
@group(0) @binding(2) var<storage, read> sh_probes: array<vec4<f32>>;

fn cascade_for(instance: u32) -> u32 {
    for (var c = 0u; c < CASCADES; c = c + 1u) {
        let p = uniforms.params[c];
        if f32(instance) >= p.x && f32(instance) < p.x + p.y {
            return c;
        }
    }
    return 0u;
}

// Clamped-cosine irradiance from one probe's coefficients (same kernel as
// `shade_cascades.wgsl::sh_irradiance`), along the sphere's world normal.
fn probe_irradiance(base: u32, n: vec3<f32>) -> vec3<f32> {
    var basis = sh_basis(n);
    let band = array<f32, 3>(PI, 2.094395, 0.785398);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let l = select(select(2u, 1u, i < 4u), 0u, i == 0u);
        e += sh_probes[base + i].xyz * (basis[i] * band[l]);
    }
    return max(e, vec3<f32>(0.0));
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) @interpolate(flat) base: u32,
}

@vertex
fn vs_main(
    @location(0) local_pos: vec3<f32>,
    @builtin(instance_index) instance: u32,
) -> VOut {
    let c = cascade_for(instance);
    let center = probe_positions[2u * instance].xyz;
    let radius = uniforms.params[c].z;
    let world = center + radius * local_pos;
    var out: VOut;
    out.clip = uniforms.clip_from_world * vec4<f32>(world, 1.0);
    // The unit icosphere is placed with a positive uniform scale and no
    // rotation, so the object-space vertex is the world-space surface normal.
    out.world_normal = local_pos;
    out.base = instance * SH_STRIDE;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let color = probe_irradiance(in.base, normalize(in.world_normal)) * PROBE_BRIGHTNESS;
    return vec4<f32>(color, 1.0);
}
