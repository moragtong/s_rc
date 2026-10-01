import re

with open("src/cascades2_5d/mod.rs", "r") as f:
    text = f.read()

# Remove the update_frame logic which depended on ClipmapState
text = re.sub(r"fn update_frame\([^}]+\) \{[^}]+\}\n", "", text)
# And replace it
new_update_frame = """
fn update_frame(
    mut frame: ResMut<Cascades2_5dFrame>,
    heightmap: Option<Res<crate::heightmap::Heightmap>>,
    sun: Option<Res<crate::scene3d::Sun>>,
) {
    frame.active = heightmap.is_some() && sun.is_some();
    if let Some(h) = heightmap {
        frame.clipmap = h.clone(); // we temporarily rename clipmap to heightmap inside Cascades2_5dFrame
    }
}
"""
text = text.replace("fn update_frame(", new_update_frame + "fn _unused_(")

with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)
