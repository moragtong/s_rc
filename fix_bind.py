import re
with open("src/heightmap_view.rs", "r") as f:
    text = f.read()

text = text.replace("#[texture(0)]", "#[texture(1)]").replace("#[sampler(1)]", "#[sampler(2)]")

with open("src/heightmap_view.rs", "w") as f:
    f.write(text)

with open("assets/shaders/heightmap_view.wgsl", "r") as f:
    wgsl = f.read()

wgsl = wgsl.replace("group(2) @binding(0)", "group(2) @binding(1)")
wgsl = wgsl.replace("group(2) @binding(1)", "group(2) @binding(2)")

with open("assets/shaders/heightmap_view.wgsl", "w") as f:
    f.write(wgsl)

