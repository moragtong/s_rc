import re

with open("assets/shaders/cascades2_5d.wgsl", "r") as f:
    text = f.read()

# Remove the SDF import and bindings
text = re.sub(r'#import s_rc::clipmap_sample::.*', '', text)
text = re.sub(r'@group\(0\) @binding\(1\) var sdf: texture_3d<f32>;\n@group\(0\) @binding\(2\) var clipmap_sampler: sampler;\n@group\(0\) @binding\(3\) var color: texture_3d<f32>;\n@group\(0\) @binding\(4\) var shadow: texture_3d<f32>;',
              r'''@group(0) @binding(1) var heightmap_tex: texture_2d<f32>;
@group(0) @binding(2) var heightmap_samp: sampler;''', text)

with open("assets/shaders/cascades2_5d.wgsl", "w") as f:
    f.write(text)
