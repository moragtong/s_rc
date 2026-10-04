// Projects each probe's merged rays onto L2 SH, one thread per probe, and
// records the probe's parallax radius: the distance of its farthest hit.
// Shading treats the probe's surroundings as a sphere of that radius.

#import s_rc::heightmap_trace::{uniforms, probe_position, sh_basis, MISS, SH_STRIDE}

@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> sh_probes: array<vec4<f32>>;
// Per cascade, per direction cell: sub^2 (direction, solid angle).
@group(1) @binding(2) var<storage, read> subsamples: array<vec4<f32>>;

fn gather(c: u32, id: u32) {
    let info = uniforms.cascades[c];
    if id >= info.size.x * info.size.y {
        return;
    }
    let rays_per_probe = info.octa_side * info.octa_side;
    let per_cell = info.sub * info.sub;

    var sh: array<vec3<f32>, 9>;
    var radius = 0.0;
    for (var r = 0u; r < rays_per_probe; r++) {
        let ray = rays[info.ray_base + id * rays_per_probe + r];
        if ray.a != MISS {
            radius = max(radius, ray.a);
        }
        for (var s = 0u; s < per_cell; s++) {
            let sub_sample = subsamples[info.sub_base + r * per_cell + s];
            var basis = sh_basis(sub_sample.xyz);
            for (var i = 0u; i < 9u; i++) {
                sh[i] += ray.rgb * (basis[i] * sub_sample.w);
            }
        }
    }

    let base = (info.probe_base + id) * SH_STRIDE;
    for (var i = 0u; i < 9u; i++) {
        sh_probes[base + i] = vec4<f32>(sh[i], 0.0);
    }
    let probe = vec2<u32>(id % info.size.x, id / info.size.x);
    sh_probes[base + 9u] = vec4<f32>(probe_position(c, probe), radius);
}

@compute @workgroup_size(64, 1, 1) fn gather_0(@builtin(global_invocation_id) id: vec3<u32>) { gather(0u, id.x); }
@compute @workgroup_size(64, 1, 1) fn gather_1(@builtin(global_invocation_id) id: vec3<u32>) { gather(1u, id.x); }
@compute @workgroup_size(64, 1, 1) fn gather_2(@builtin(global_invocation_id) id: vec3<u32>) { gather(2u, id.x); }
@compute @workgroup_size(64, 1, 1) fn gather_3(@builtin(global_invocation_id) id: vec3<u32>) { gather(3u, id.x); }
