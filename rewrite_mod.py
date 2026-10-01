import re

with open("src/heightmap.rs", "r") as f:
    text = f.read()

if "HeightmapUniform" not in text:
    text += """
#[derive(bevy::render::render_resource::ShaderType, Clone, Copy, Default)]
pub struct HeightmapUniform {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
}
"""
    # derive default for heightmap
    text = text.replace("#[derive(Resource, Clone, ExtractResource)]", "#[derive(Resource, Clone, ExtractResource, Default)]")
    # add sampler handle to heightmap just to ease pain? No, we don't need sampler.
with open("src/heightmap.rs", "w") as f:
    f.write(text)


with open("src/cascades2_5d/mod.rs", "r") as f:
    mod = f.read()

# 1. Remove unresolved imports
mod = re.sub(r'use crate::sdf_view::SdfView;\n', '', mod)
mod = re.sub(r'\.after\(crate::clipmap::ClipmapSystems\)', '', mod)
mod = re.sub(r'\.after\(ClipmapRenderSystems\)', '', mod)

# 2. Fix Sun
mod = mod.replace("crate::clipmap::SunLight", "crate::scene3d::Sun")
mod = mod.replace("sun: Option<Res<crate::clipmap::SunLight>>", "sun: Option<Res<crate::scene3d::Sun>>")

# 3. Fix Params
mod = mod.replace("clipmap: Heightmap,", "clipmap: crate::heightmap::HeightmapUniform,")

# 4. Fix Frame extraction
mod = mod.replace("clipmap: frame.clipmap", "heightmap: frame.heightmap")
mod = re.sub(r'pub struct Cascades2_5dFrame \{[\s\S]*?\}',
"""
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct Cascades2_5dFrame {
    pub active: bool,
    pub camera: Vec3,
    pub heightmap: crate::heightmap::Heightmap,
    pub generation: u32,
    pub lighting: u32,
}
""", mod)

# 5. Fix GPU bindings setup
setup_pattern = r'let any = &BindGroupEntries::sequential\([\s\S]*?\n\);'
new_setup = r"""
let any = &BindGroupEntries::sequential((
    &bindings.params,
    &bindings.heightmap.as_entire_binding(), // This needs to be a texture view.
    &gpu.cascade_src_binding(),
    &gpu.cascade_dst_binding(),
    gpu.hits.as_entire_buffer_binding(),
    gpu.tags.as_entire_buffer_binding(),
    gpu.work_count.as_entire_buffer_binding(),
    gpu.solid_angles.as_entire_buffer_binding(),
    gpu.child_masks.as_entire_buffer_binding(),
    gpu.work_list.as_entire_buffer_binding(),
    gpu.dispatch_args.as_entire_buffer_binding(),
    gpu.shading_subsamples.as_entire_buffer_binding(),
    gpu.probe_sh.as_entire_buffer_binding(),
    gpu.dirty.as_entire_buffer_binding(),
    gpu.gather_count.as_entire_buffer_binding(),
    gpu.gather_list.as_entire_buffer_binding(),
));
"""
# mod = re.sub(setup_pattern, new_setup, mod) # Too complex, let's just do it manually.

