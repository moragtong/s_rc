// The shadow grid: sun visibility for one region of the SDF grid, for
// voxels near surfaces, at the nearest surface point. Visibility is kept
// separate from surface color (clipmap_compose.wgsl) so it can be re-marched
// on its own, e.g. double-buffered for time of day. Runs after geometry for
// all of this frame's regions is composed, since a march reads beyond its
// region.

#import s_rc::clipmap_sample::{Clipmap, sample_sdf, sun_visibility}

struct ShadowParams {
    clipmap: Clipmap,
    region_min: vec3<i32>,
    sun_k: f32,
    region_size: vec3<u32>,
    sun_toward: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: ShadowParams;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var clipmap_sampler: sampler;
@group(0) @binding(3) var shadow_out: texture_storage_3d<r16float, write>;

// Same band as the color in clipmap_compose.wgsl.
const SURFACE_BAND: f32 = 2.5;
// Start the march this far off the surface: past the hit threshold (a
// quarter voxel), as for camera pixels.
const START_OFFSET: f32 = 0.07;

fn sample_at(p: vec3<f32>) -> f32 {
    return sample_sdf(params.clipmap, sdf, clipmap_sampler, p);
}

@compute @workgroup_size(4, 4, 4)
fn shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.region_size) {
        return;
    }
    let v = params.region_min + vec3<i32>(id);
    let res = i32(params.clipmap.resolution);
    let texel = (v % res + res) % res;
    let voxel = params.clipmap.voxel;
    let pos = (vec3<f32>(v) + 0.5) * voxel;

    var visibility = 1.0;
    let d = sample_at(pos);
    if abs(d) < SURFACE_BAND * voxel {
        let e = 0.5 * voxel;
        let n = normalize(vec3(
            sample_at(pos + vec3(e, 0.0, 0.0)) - sample_at(pos - vec3(e, 0.0, 0.0)),
            sample_at(pos + vec3(0.0, e, 0.0)) - sample_at(pos - vec3(0.0, e, 0.0)),
            sample_at(pos + vec3(0.0, 0.0, e)) - sample_at(pos - vec3(0.0, 0.0, e)),
        ));
        // Every near-surface voxel is marched, including ones whose normal
        // here faces away from the sun: the color takes its normal from the
        // nearest instance, this from the composed field, and at junctions
        // they differ. Skipping those (visibility 1, "no color anyway") put
        // the floor's unshadowed sunlit color, times 1, along the base of
        // walls facing away from the sun: light leaking into room corners.
        let surface = pos - n * d;
        visibility = sun_visibility(
            params.clipmap, sdf, clipmap_sampler,
            surface + n * START_OFFSET, params.sun_toward, params.sun_k,
        );
    }
    textureStore(shadow_out, texel, vec4(visibility, 0.0, 0.0, 0.0));
}
