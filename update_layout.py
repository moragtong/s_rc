import re

with open("src/cascades2_5d/layout.rs", "r") as f:
    text = f.read()

text = text.replace("pub size: u32,", "pub size: bevy::math::UVec2,")

# Update rays function
text = text.replace("(self.size as u64).pow(3) * self.rays_per_probe() as u64",
                    "(self.size.x as u64 * self.size.y as u64) * self.rays_per_probe() as u64")

# Update allocate logic
text = text.replace("tag_base += (w.size as u64).pow(3);",
                    "tag_base += (w.size.x as u64 * w.size.y as u64);")
text = text.replace("let window = |radius: f32, s: f32| 2 * ((radius / s).ceil() as u32 + 2);",
                    "let window = |radius: f32, s: f32| 2 * ((radius / s).ceil() as u32 + 2);")

# We need to change `size: window(radius, s)` to `size: UVec2::splat(window(radius, s))`
text = text.replace("size: window(radius, s),", "size: bevy::math::UVec2::splat(window(radius, s)),")

# 3D bilinear fix -> 2D bilinear fix traces one per bilinear child
# Update segments() function
text = text.replace("if self.child.is_some() { 8 } else { 1 }", "if self.child.is_some() { 4 } else { 1 }")

with open("src/cascades2_5d/layout.rs", "w") as f:
    f.write(text)
