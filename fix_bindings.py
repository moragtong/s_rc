import re
with open("src/cascades2_5d/mod.rs", "r") as f:
    text = f.read()

# Replace BindGroupEntries setup
p1 = r'let any = &BindGroupEntries::sequential\(\(\n\s*&bindings\.params,\n.*?\n.*?\n.*?\n.*?\n'
# we will just replace the specific texture paths.
text = text.replace("&textures.sdf,", "&gpu.heightmap_view,")
text = text.replace("&textures.sampler,", "// sampler gone")
text = text.replace("&textures.color,", "// color gone")
text = text.replace("&textures.shadow,", "// shadow gone")

text = text.replace("sun.is_some_and(|s| s.is_changed())", "sun.is_some()")

text = text.replace("window: own.size,", "window: own.size.x,")
text = text.replace("child_window: child.size,", "child_window: child.size.x,")
text = text.replace("test_slot,", "test_slot: test_slot.x,")

text = text.replace("(top.size as u64).pow(3)", "(top.size.x as u64 * top.size.y as u64)")
text = text.replace("windows[n].size.pow(3)", "(windows[n].size.x * windows[n].size.y)")
text = text.replace("windows[n].size.x.pow(3)", "(windows[n].size.x * windows[n].size.y)")
text = text.replace("IVec3::splat(top.size as i32)", "IVec3::new(top.size.x as i32, 0, top.size.y as i32)")

# we need to inject the heightmap view into Gpu struct or locally.
# The heightmap is just an image. We can grab the view from RenderAssets<GpuImage>.
extract_frame = """
fn extract_frame(
    mut frame: ResMut<Cascades2_5dFrame>,
    heightmap: Option<Extract<Res<crate::heightmap::Heightmap>>>,
    sun: Option<Extract<Res<crate::scene3d::Sun>>>,
    camera: Extract<Query<&GlobalTransform, With<Camera3d>>>,
) {
"""
text = re.sub(r'fn update_frame\([\s\S]*?\{', extract_frame, text)

with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)

