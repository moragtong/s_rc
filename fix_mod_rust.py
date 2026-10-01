import re

with open("src/heightmap.rs", "r") as f:
    text = f.read()

# Add HeightmapUniform
uniform_code = """
#[derive(ShaderType, Clone, Copy)]
pub struct HeightmapUniform {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
}

impl HeightmapUniform {
    pub fn new(h: &Heightmap) -> Self {
        Self {
            min_bounds: h.min_bounds,
            max_bounds: h.max_bounds,
            resolution: h.resolution,
        }
    }
}
"""
text = text.replace("pub texture: Handle<Image>,", "pub texture: Handle<Image>,\n    pub sampler: Handle<Sampler>,")

# Find setup_heightmap
setup_pattern = r"(let texture = images\.add\(image\);)"
setup_repl = r"\1\n    let sampler = app.world().resource::<RenderDevice>().create_sampler(&SamplerDescriptor::default());" # Wait, cannot create sampler like this in setup_heightmap easily. Instead just extract linear_sampler.

# let's just make it simple: replace the whole file? No, just add derivation to Sun and create HeightmapUniform.

with open("src/scene3d.rs", "r") as f:
    sc = f.read()
sc = sc.replace("pub struct Sun {", "#[derive(Resource)]\npub struct Sun {")
with open("src/scene3d.rs", "w") as f:
    f.write(sc)

