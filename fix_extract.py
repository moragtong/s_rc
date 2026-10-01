import re
with open("src/heightmap.rs", "r") as f:
    text = f.read()

pattern = r"let Some\(render_app\).*?\n\s*render_app\s*\.add_plugins\(ExtractResourcePlugin::<Heightmap>::default\(\)\);"
text = re.sub(pattern, "app.add_plugins(ExtractResourcePlugin::<Heightmap>::default());", text)

with open("src/heightmap.rs", "w") as f:
    f.write(text)

