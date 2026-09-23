// Recomposes one region of one clipmap tier from the baked per-mesh SDF
// grids: min over instances of each grid's trilinearly sampled distance.
// World voxel v is stored in texel v mod resolution (toroidal).

struct Instance {
    world_to_local: mat4x4<f32>,
    grid_min: vec3<f32>,
    voxel: f32,
    dims: vec3<u32>,
    offset: u32,
}

struct ComposeParams {
    region_min: vec3<i32>,
    instance_count: u32,
    region_size: vec3<u32>,
    resolution: u32,
    voxel: f32,
}

@group(0) @binding(0) var<uniform> params: ComposeParams;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var tier: texture_storage_3d<r16float, write>;

fn grid_value(inst: Instance, cell: vec3<i32>) -> f32 {
    let c = vec3<u32>(clamp(cell, vec3(0), vec3<i32>(inst.dims) - 1));
    return values[inst.offset + c.x + inst.dims.x * (c.y + inst.dims.y * c.z)];
}

// Trilinear inside the grid (values live at voxel centers). Outside it, the
// distance to the grid's sample bounds plus the value there: an upper bound
// on the true distance, exact when the nearest surface is straight through
// the nearest bound point, which the bake's padding makes the common case.
fn sample_instance(inst: Instance, p: vec3<f32>) -> f32 {
    let lo = inst.grid_min + 0.5 * inst.voxel;
    let hi = inst.grid_min + (vec3<f32>(inst.dims) - 0.5) * inst.voxel;
    let q = clamp(p, lo, hi);
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
    for (var i = 0u; i < params.instance_count; i++) {
        let inst = instances[i];
        let local = (inst.world_to_local * vec4(pos, 1.0)).xyz;
        d = min(d, sample_instance(inst, local));
    }
    textureStore(tier, texel, vec4(d, 0.0, 0.0, 0.0));
}
