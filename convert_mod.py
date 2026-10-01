import re

with open("cascades3d_mod.rs", "r") as f:
    text = f.read()

# Rename struct and plugin
text = text.replace("Cascades3d", "Cascades2_5d")
text = text.replace("cascades3d", "cascades2_5d")
text = text.replace("CASCADES3D", "CASCADES2_5D")

# Heightmap replacements
text = text.replace("crate::clipmap::{ClipmapCamera, ClipmapRenderSystems, ClipmapState, ClipmapTextures, ClipmapUniform}", 
                    "crate::heightmap::Heightmap")
text = text.replace("ClipmapTextures", "Heightmap")
text = text.replace("ClipmapUniform", "Heightmap")
text = text.replace("clipmap: frame.heightmap.clone(),", "heightmap: frame.heightmap.clone(),")

# Replace ClipmapState with Heightmap in structs
text = re.sub(r'ClipmapState', 'Heightmap', text)

# update_frame needs proper Heightmap injection
update_frame = """fn update_frame(
    mut frame: ResMut<Cascades2_5dFrame>,
    heightmap: Option<Res<Heightmap>>,
    sun: Option<Res<crate::scene3d::Sun>>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
) {
    let (Some(heightmap), Ok(camera)) = (heightmap, camera.get_single()) else { return };
    if sun.is_some_and(|s| s.is_changed()) {
        frame.lighting += 1;
    }
    frame.active = true;
    frame.camera = camera.translation();
    frame.heightmap = heightmap.clone();
    frame.generation = 1;
}"""
text = re.sub(r'fn update_frame\([\s\S]*?\n\}', update_frame, text)

# Frame and State
text = text.replace("pub clipmap: Heightmap,", "pub heightmap: Heightmap,")

# Binding adjustments. Remove SDF, color, shadow 3D textures. Add heightmap_tex, heightmap_samp.
# old binding entries:
# 1: &textures.sdf
# 2: &textures.sampler
# 3: &textures.color
# 4: &textures.shadow
text = re.sub(r'let any = &BindGroupEntries::sequential\(\(\n.*?\n.*?\n.*?\n.*?\n.*?\n', 
              r'''let any = &BindGroupEntries::sequential((
                &bindings.params,
                &bindings.heightmap.texture,
                &bindings.heightmap.sampler,
''', text)

with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)

