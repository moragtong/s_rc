import re

# cascades3d.wgsl modifications
with open('assets/shaders/cascades3d.wgsl', 'r') as f:
    text = f.read()

text = text.replace(
    'child_octa_side: u32,\n    // This pass\'s',
    'child_octa_side: u32,\n    // This pass\'s'
)
text = text.replace(
    'window: u32,\n    radius: f32,',
    'window: u32,\n    hash_size: u32,\n    radius: f32,'
)
text = text.replace(
    'child_window: u32,\n    child_radius: f32,',
    'child_window: u32,\n    child_hash_size: u32,\n    child_radius: f32,'
)
text = text.replace(
    '@group(0) @binding(18) var<storage, read_write> gather_list: array<u32>;',
    '@group(0) @binding(18) var<storage, read_write> gather_list: array<u32>;\n@group(0) @binding(19) var<storage, read_write> locks: array<atomic<u32>>;'
)

unpack_tag = """
fn unpack_tag(tag: u32) -> vec3<i32> {
    let t = tag - 1u;
    let x = i32(t & 1023u) - 512;
    let y = i32((t >> 10u) & 1023u) - 512;
    let z = i32((t >> 20u) & 1023u) - 512;
    return vec3<i32>(x, y, z);
}
"""
text = text.replace('fn pack_tag(world: vec3<i32>) -> u32 {', unpack_tag + '\nfn pack_tag(world: vec3<i32>) -> u32 {')

slot_world = """
fn slot_world(slot: u32, spacing: f32, window: u32, hash_size: u32, tag_base: u32) -> vec3<i32> {
    if hash_size > 0u {
        let tag = tags[tag_base + slot];
        if tag != 0u {
            return unpack_tag(tag);
        }
        return vec3<i32>(2000, 2000, 2000); // Invalid far away
    }
    let size = i32(window);
    let s = vec3<i32>(vec3(slot % window, (slot / window) % window, slot / (window * window)));
    let origin = vec3<i32>(floor(params.camera / spacing)) - size / 2;
    return origin + ((s - origin) % size + size) % size;
}"""
text = text.replace(
    'fn slot_world(slot: u32, spacing: f32, window: u32) -> vec3<i32> {\n    let size = i32(window);\n    let s = vec3<i32>(vec3(slot % window, (slot / window) % window, slot / (window * window)));\n    let origin = vec3<i32>(floor(params.camera / spacing)) - size / 2;\n    return origin + ((s - origin) % size + size) % size;\n}',
    slot_world
)

world_slot = """
fn world_slot(world: vec3<i32>, window: u32, hash_size: u32, tag_base: u32) -> u32 {
    if hash_size > 0u {
        let target = pack_tag(world);
        // Simple fast spatial hash
        let h = (u32(world.x) * 73856093u ^ u32(world.y) * 19349663u ^ u32(world.z) * 83492791u) % hash_size;
        for (var i = 0u; i < 16u; i++) {
            let slot = (h + i) % hash_size;
            if tags[tag_base + slot] == target {
                return slot;
            }
        }
        return 0xFFFFFFFFu;
    }
    let size = i32(window);
    let s = vec3<u32>((world % size + size) % size);
    return s.x + window * (s.y + window * s.z);
}"""
text = text.replace(
    'fn world_slot(world: vec3<i32>, window: u32) -> u32 {\n    let size = i32(window);\n    let s = vec3<u32>((world % size + size) % size);\n    return s.x + window * (s.y + window * s.z);\n}',
    world_slot
)

# Update select_probes
select_probes = """
@compute @workgroup_size(64, 1, 1)
fn select_probes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = linear_index(gid, groups);
    if idx >= params.window * params.window * params.window { return; }
    
    let size = i32(params.window);
    let s = vec3<i32>(vec3(idx % params.window, (idx / params.window) % params.window, idx / (params.window * params.window)));
    let origin = vec3<i32>(floor(params.camera / params.spacing)) - size / 2;
    let world = origin + ((s - origin) % size + size) % size;
    
    if !probe_valid(world, params.spacing, params.radius) { return; }
    
    if params.hash_size > 0u {
        let center = probe_center(world, params.spacing);
        let empty = in_grid(params.clipmap, center) && sample_sdf(params.clipmap, sdf, clipmap_sampler, center) > max(hit_eps(params.clipmap), 0.03125);
        if empty { return; }
        
        let target = pack_tag(world);
        let h = (u32(world.x) * 73856093u ^ u32(world.y) * 19349663u ^ u32(world.z) * 83492791u) % params.hash_size;
        var found = 0xFFFFFFFFu;
        for (var i = 0u; i < 16u; i++) {
            let slot = (h + i) % params.hash_size;
            if tags[params.tag_base + slot] == target {
                found = slot;
                break;
            }
        }
        if found == 0xFFFFFFFFu {
            for (var i = 0u; i < 16u; i++) {
                let slot = (h + i) % params.hash_size;
                let old_tag = tags[params.tag_base + slot];
                let owner_world = unpack_tag(old_tag);
                if old_tag == 0u || !probe_valid(owner_world, params.spacing, params.radius) {
                    if atomicCompareExchangeWeak(&locks[params.tag_base + slot], 0u, 1u).exchanged {
                        tags[params.tag_base + slot] = target;
                        work_list[params.tag_base + atomicAdd(&work_count[params.pass_index], 1u)] = slot;
                        atomicStore(&locks[params.tag_base + slot], 0u);
                        break;
                    }
                }
            }
        }
    } else {
        if tags[params.tag_base + idx] != pack_tag(world) {
            work_list[params.tag_base + atomicAdd(&work_count[params.pass_index], 1u)] = idx;
        }
    }
}
"""
text = re.sub(r'@compute @workgroup_size\(64, 1, 1\)\nfn select_probes.*?\}', select_probes, text, flags=re.DOTALL)

# Update mark_dirty
text = text.replace(
    'let world = slot_world(slot, params.spacing, params.window);',
    'let world = slot_world(slot, params.spacing, params.window, params.hash_size, params.tag_base);'
)
text = text.replace(
    'if slot >= params.window * params.window * params.window {\n        return;\n    }',
    'let capacity = select(params.window * params.window * params.window, params.hash_size, params.hash_size > 0u);\n    if slot >= capacity {\n        return;\n    }'
)
text = text.replace(
    'let child_slot = world_slot(t.base + corner(k), params.child_window);',
    'let child_slot = world_slot(t.base + corner(k), params.child_window, params.child_hash_size, params.child_tag_base);\n            if child_slot == 0xFFFFFFFFu { continue; }'
)


# Update trace & gather & commit slot_world arguments
text = text.replace(
    'let world = slot_world(slot, params.spacing, params.window);',
    'let world = slot_world(slot, params.spacing, params.window, params.hash_size, params.tag_base);'
)
text = text.replace(
    'let child_slot = world_slot(child, params.child_window);',
    'let child_slot = world_slot(child, params.child_window, params.child_hash_size, params.child_tag_base);'
)
text = text.replace(
    '&& tags[params.child_tag_base + child_slot] == pack_tag(child);',
    '&& child_slot != 0xFFFFFFFFu && tags[params.child_tag_base + child_slot] == pack_tag(child);'
)

commit = """
@compute @workgroup_size(64, 1, 1)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    if params.hash_size > 0u {
        return; // Allocation and tag commit happen atomically in select_probes
    }
    let id = vec3(linear_index(gid, groups), 0u, 0u);
    let slots = params.window * params.window * params.window;
    if id.x >= slots {
        return;
    }
    let world = slot_world(id.x, params.spacing, params.window, 0u, params.tag_base);
    tags[params.tag_base + id.x] = select(0u, pack_tag(world), probe_valid(world, params.spacing, params.radius));
}
"""
text = re.sub(r'@compute @workgroup_size\(64, 1, 1\)\nfn commit.*?\}', commit, text, flags=re.DOTALL)


with open('assets/shaders/cascades3d.wgsl', 'w') as f:
    f.write(text)


# sdf_view.wgsl modifications
with open('assets/shaders/sdf_view.wgsl', 'r') as f:
    text = f.read()

text = text.replace(
    'voxel: f32,\n    // Cascade 0\'s probe window',
    'voxel: f32,\n    c0_window: u32,\n    c0_hash_size: u32,\n    c0_radius: f32,\n    c0_tag_base: u32,\n    spacing0: f32,\n    sun_k: f32,\n    shade_debug: u32,\n    sun_toward: vec3<f32>,\n    pad1: u32,\n    sun_irradiance: vec3<f32>,\n    pad2: u32,\n    // Cascade 0\'s probe window'
)
text = text.replace(
    'c0_window: u32,\n    c0_radius: f32,\n    c0_tag_base: u32,\n    spacing0: f32,\n    sun_toward: vec3<f32>,\n    sun_k: f32,\n    sun_irradiance: vec3<f32>,\n    // DEBUG (RC3D_SHADE_DEBUG): 1 = no probe dropout, 2 = nearest probe.\n    shade_debug: u32,',
    ''
)

c0_slot = """
fn c0_slot(world: vec3<i32>) -> u32 {
    if view.c0_hash_size > 0u {
        let target = pack_tag(world);
        let h = (u32(world.x) * 73856093u ^ u32(world.y) * 19349663u ^ u32(world.z) * 83492791u) % view.c0_hash_size;
        for (var i = 0u; i < 16u; i++) {
            let slot = (h + i) % view.c0_hash_size;
            if tags[view.c0_tag_base + slot] == target {
                return slot;
            }
        }
        return 0xFFFFFFFFu;
    }
    let size = i32(view.c0_window);
    let s = vec3<u32>((world % size + size) % size);
    return s.x + view.c0_window * (s.y + view.c0_window * s.z);
}
"""
text = re.sub(r'fn c0_slot\(world: vec3<i32>\) -> u32 \{.*?\}', c0_slot, text, flags=re.DOTALL)

text = text.replace(
    '&& tags[view.c0_tag_base + slot] == pack_tag(world);',
    '&& slot != 0xFFFFFFFFu;'
)

with open('assets/shaders/sdf_view.wgsl', 'w') as f:
    f.write(text)

