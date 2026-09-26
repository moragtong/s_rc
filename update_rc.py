import re
import os

# 1. Update layout.rs
with open('src/cascades3d/layout.rs', 'r') as f:
    text = f.read()

text = text.replace(
    'pub size: u32,\n    /// Probes within this distance',
    'pub size: u32,\n    pub hash_size: u32,\n    /// Probes within this distance'
)
text = text.replace(
    'pub fn rays(&self) -> u64 {\n        (self.size as u64).pow(3)',
    'pub fn capacity(&self) -> u32 {\n        if self.hash_size > 0 { self.hash_size } else { self.size.pow(3) }\n    }\n\n    pub fn rays(&self) -> u64 {\n        self.capacity() as u64'
)
text = text.replace(
    'tag_base += (w.size as u64).pow(3);\n            hit_base += w.rays() * w.hit_words() as u64;',
    'tag_base += w.capacity() as u64;\n            hit_base += w.rays() * w.hit_words() as u64;'
)
text = text.replace(
    'size: window(radius, s),\n                radius,',
    'size: window(radius, s),\n                hash_size: if n == 0 { 131_072 } else { 0 },\n                radius,'
)
with open('src/cascades3d/layout.rs', 'w') as f:
    f.write(text)


# 2. Update cascades3d/mod.rs
with open('src/cascades3d/mod.rs', 'r') as f:
    text = f.read()

text = text.replace('pub tags: Buffer,\n    /// Per slot', 'pub tags: Buffer,\n    locks: Buffer,\n    /// Per slot')
text = text.replace('tags: storage(&render_device, "cascades3d_tags", layout.tag_words * 4),', 'tags: storage(&render_device, "cascades3d_tags", layout.tag_words * 4),\n        locks: storage(&render_device, "cascades3d_locks", layout.tag_words * 4),')
text = text.replace('storage_buffer_sized(false, None),\n                storage_buffer_sized(false, None),\n            ),', 'storage_buffer_sized(false, None),\n                storage_buffer_sized(false, None),\n                storage_buffer_sized(false, None),\n            ),')

text = text.replace(
    'gpu.gather_list.as_entire_buffer_binding(),\n            )),',
    'gpu.gather_list.as_entire_buffer_binding(),\n                gpu.locks.as_entire_buffer_binding(),\n            )),'
)
text = text.replace(
    'child_window: u32,\n    child_radius: f32,',
    'child_window: u32,\n    child_hash_size: u32,\n    child_radius: f32,'
)
text = text.replace(
    'window: u32,\n    radius: f32,',
    'window: u32,\n    hash_size: u32,\n    radius: f32,'
)
text = text.replace(
    'window: own.size,\n            radius: own.radius,',
    'window: own.size,\n            hash_size: own.hash_size,\n            radius: own.radius,'
)
text = text.replace(
    'child_window: child.size,\n            child_radius: child.radius,',
    'child_window: child.size,\n            child_hash_size: child.hash_size,\n            child_radius: child.radius,'
)
text = text.replace('top.size as u64).pow(3)', 'top.capacity() as u64')
text = text.replace('dispatch_linear(&mut pass, w.size.pow(3));', 'dispatch_linear(&mut pass, w.capacity());')
# Wait, select_probes MUST dispatch w.size.pow(3) still! Let's revert that part manually if replaced:
text = text.replace('let span = diagnostics.pass_span(&mut pass, SELECT_SPAN);\n        pass.set_pipeline(select);\n        for (w, offset) in windows.iter().zip(&offsets) {\n            pass.set_bind_group(0, any, &[*offset]);\n            dispatch_linear(&mut pass, w.capacity());\n        }', 'let span = diagnostics.pass_span(&mut pass, SELECT_SPAN);\n        pass.set_pipeline(select);\n        for (w, offset) in windows.iter().zip(&offsets) {\n            pass.set_bind_group(0, any, &[*offset]);\n            dispatch_linear(&mut pass, w.size.pow(3));\n        }')
# And for mark_dirty: 'dispatch_linear(&mut pass, windows[n].size.pow(3));'
text = text.replace('dispatch_linear(&mut pass, windows[n].size.pow(3));', 'dispatch_linear(&mut pass, windows[n].capacity());')

with open('src/cascades3d/mod.rs', 'w') as f:
    f.write(text)


# 3. Update sdf_view.rs
with open('src/sdf_view.rs', 'r') as f:
    text = f.read()

text = text.replace(
    'voxel: f32,\n    c0_window: u32,\n    c0_radius: f32,\n    c0_tag_base: u32,\n    spacing0: f32,\n    sun_toward: Vec3,\n    sun_k: f32,\n    sun_irradiance: Vec3,\n    shade_debug: u32,',
    'voxel: f32,\n    c0_window: u32,\n    c0_hash_size: u32,\n    c0_radius: f32,\n    c0_tag_base: u32,\n    spacing0: f32,\n    sun_k: f32,\n    shade_debug: u32,\n    sun_toward: Vec3,\n    _pad1: u32,\n    sun_irradiance: Vec3,\n    _pad2: u32,'
)
text = text.replace(
    'voxel: VOXEL,\n        c0_window: c0.size,\n        c0_radius: c0.radius,\n        c0_tag_base: c0.tag_base,\n        spacing0: SPACING0,\n        shade_debug',
    'voxel: VOXEL,\n        c0_window: c0.size,\n        c0_hash_size: if c0.size > 0 { 131_072 } else { 0 },\n        c0_radius: c0.radius,\n        c0_tag_base: c0.tag_base,\n        spacing0: SPACING0,\n        shade_debug'
)
text = text.replace(
    'sun_irradiance: sun.irradiance,\n    };',
    'sun_irradiance: sun.irradiance,\n        _pad1: 0,\n        _pad2: 0,\n    };'
)
with open('src/sdf_view.rs', 'w') as f:
    f.write(text)


print("Python script generated!")
