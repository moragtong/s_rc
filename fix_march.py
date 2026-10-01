import re

with open("assets/shaders/cascades2_5d.wgsl", "r") as f:
    text = f.read()

# Just stubbing march_fraction to compile for now
new_march = """
fn march_fraction(seg: Segment) -> f32 {
    let delta = seg.b - seg.a;
    let len = length(delta);
    let dir = delta / len;
    
    // TODO: 2D DDA over the heightmap
    
    return 1.0;
}
"""
text = re.sub(r'fn march_fraction\(seg: Segment\) -> f32 \{[\s\S]*?\}', new_march, text)

with open("assets/shaders/cascades2_5d.wgsl", "w") as f:
    f.write(text)
