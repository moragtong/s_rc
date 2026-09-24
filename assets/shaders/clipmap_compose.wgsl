// Recomposes one region of the SDF grid from the baked per-mesh SDF
// grids: min over instances of each grid's trilinearly sampled distance.
// Near surfaces it also writes the nearest instance's albedo and the
// unshadowed surface color, albedo / pi * sun irradiance * cos, with the
// normal from that instance's own grid (sharper than the composed field's).
// Sun visibility is a separate pass (clipmap_shadow.wgsl).
// World voxel v is stored in texel v mod resolution (toroidal).

struct Instance {
    world_to_local: mat4x4<f32>,
    grid_min: vec3<f32>,
    voxel: f32,
    dims: vec3<u32>,
    offset: u32,
    albedo: vec4<f32>,
    // The mesh's local bounds.
    mesh_min: vec3<f32>,
    mesh_max: vec3<f32>,
}

struct ComposeParams {
    region_min: vec3<i32>,
    instance_count: u32,
    region_size: vec3<u32>,
    resolution: u32,
    sun_toward: vec3<f32>,
    voxel: f32,
    sun_irradiance: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: ComposeParams;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var sdf_out: texture_storage_3d<r16float, write>;
@group(0) @binding(4) var albedo_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(5) var color_out: texture_storage_3d<rgba16float, write>;

const PI: f32 = 3.14159265359;
// Voxels within this many voxels of a surface get albedo and color: enough
// that trilinear lookups only mix voxels that have them. Lookups sit up to
// about 0.75 voxel off a surface (half a voxel out, plus the hit threshold)
// and reach one voxel further; at 1.5, the second voxel row out held the
// defaults (color 0, visibility 1), blended in by how far off a march
// stopped: rings, and sunlight in shadowed rooms.
const SURFACE_BAND: f32 = 2.5;

fn grid_value(inst: Instance, cell: vec3<i32>) -> f32 {
    let c = vec3<u32>(clamp(cell, vec3(0), vec3<i32>(inst.dims) - 1));
    return values[inst.offset + c.x + inst.dims.x * (c.y + inst.dims.y * c.z)];
}

// Trilinear inside the grid (values live at voxel centers). Outside it, the
// distance to the mesh's bounds: never more
// than the true distance, so sphere tracing can't step past a surface, and
// exact for boxes. (It was the distance to the grid's edge plus the value
// there, an upper bound that overshot by up to 14% diagonally past corners:
// 9.5 cm outside the ceiling's corners.)
fn sample_instance(inst: Instance, p: vec3<f32>) -> f32 {
    let lo = inst.grid_min + 0.5 * inst.voxel;
    let hi = inst.grid_min + (vec3<f32>(inst.dims) - 0.5) * inst.voxel;
    if any(p < lo) || any(p > hi) {
        return length(max(max(inst.mesh_min - p, p - inst.mesh_max), vec3(0.0)));
    }
    let q = p;
    let g = (q - inst.grid_min) / inst.voxel - 0.5;
    let c = vec3<i32>(floor(g));
    let f = g - floor(g);
    let x00 = mix(grid_value(inst, c), grid_value(inst, c + vec3(1, 0, 0)), f.x);
    let x10 = mix(grid_value(inst, c + vec3(0, 1, 0)), grid_value(inst, c + vec3(1, 1, 0)), f.x);
    let x01 = mix(grid_value(inst, c + vec3(0, 0, 1)), grid_value(inst, c + vec3(1, 0, 1)), f.x);
    let x11 = mix(grid_value(inst, c + vec3(0, 1, 1)), grid_value(inst, c + vec3(1, 1, 1)), f.x);
    let d = mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
    return length(p - q) + d;
}

@compute @workgroup_size(4, 4, 4)
fn compose(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.region_size) {
        return;
    }
    let v = params.region_min + vec3<i32>(id);
    let res = i32(params.resolution);
    let texel = (v % res + res) % res;
    let pos = (vec3<f32>(v) + 0.5) * params.voxel;

    var d = 1e4;
    var nearest = 0u;
    for (var i = 0u; i < params.instance_count; i++) {
        let inst = instances[i];
        let local = (inst.world_to_local * vec4(pos, 1.0)).xyz;
        let di = sample_instance(inst, local);
        if di < d {
            d = di;
            nearest = i;
        }
    }
    textureStore(sdf_out, texel, vec4(d, 0.0, 0.0, 0.0));

    var albedo = vec3(0.0);
    var color = vec3(0.0);
    if abs(d) < SURFACE_BAND * params.voxel {
        let inst = instances[nearest];
        let local = (inst.world_to_local * vec4(pos, 1.0)).xyz;
        let e = 0.5 * inst.voxel;
        let local_normal = vec3(
            sample_instance(inst, local + vec3(e, 0.0, 0.0)) - sample_instance(inst, local - vec3(e, 0.0, 0.0)),
            sample_instance(inst, local + vec3(0.0, e, 0.0)) - sample_instance(inst, local - vec3(0.0, e, 0.0)),
            sample_instance(inst, local + vec3(0.0, 0.0, e)) - sample_instance(inst, local - vec3(0.0, 0.0, e)),
        );
        // Normals transform by the inverse transpose of local_to_world,
        // i.e. the transpose of world_to_local.
        let n = normalize((transpose(inst.world_to_local) * vec4(local_normal, 0.0)).xyz);
        albedo = inst.albedo.rgb;
        // Only voxels in empty space carry a face's light. Inside a wall, the
        // nearest face may be the other side's (e.g. the sunlit outside), and
        // a lookup at the inner face would blend that in: a leak.
        if d >= 0.0 {
            color = albedo / PI * params.sun_irradiance * max(dot(n, params.sun_toward), 0.0);
        }
    }
    textureStore(albedo_out, texel, vec4(albedo, 1.0));
    textureStore(color_out, texel, vec4(color, 1.0));
}
