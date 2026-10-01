import re

with open("src/cascades2_5d/mod.rs", "r") as f:
    text = f.read()

# 1. replace imports
text = text.replace("use crate::clipmap::{ClipmapCamera, ClipmapRenderSystems, ClipmapState, ClipmapTextures, ClipmapUniform};", 
                    "use crate::heightmap::Heightmap;")
text = text.replace("use crate::sdf_view::SdfView;", "")

text = text.replace("ClipmapUniform", "Heightmap")
text = text.replace("ClipmapState", "Heightmap")

text = text.replace("clipmap: Heightmap,", "heightmap: Heightmap,")
text = text.replace("frame.clipmap", "frame.heightmap")
text = text.replace("clipmap: frame.heightmap", "heightmap: frame.heightmap.clone()")


# 2. update_frame
update_frame_sig = r"""fn update_frame\(
    mut frame: ResMut<Cascades2_5dFrame>,
    view: Option<Res<SdfView>>,
    state: Res<Heightmap>,
    sun: Option<Res<super::scene3d::Sun>>,
    camera: Query<&GlobalTransform, With<ClipmapCamera>>,
\) \{
    let \(Some\(view\), Ok\(camera\)\) = \(view, camera.get_single\(\)\) else \{ return \};
    if sun.is_some_and\(\|s\| s.is_changed\(\)\) \{
        frame.lighting \+= 1;
    \}
    frame.active = view.enabled && view.mode >= LIGHTING_MODE;
    frame.camera = camera.translation\(\);
    frame.heightmap = Heightmap::from_state\(&state\);
    frame.generation = state.generation;
\}"""

new_update_frame = """fn update_frame(
    mut frame: ResMut<Cascades2_5dFrame>,
    heightmap: Option<Res<Heightmap>>,
    sun: Option<Res<super::scene3d::Sun>>,
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
text = re.sub(r"fn update_frame\([^}]+\} *\n\}", new_update_frame, text)

with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)
