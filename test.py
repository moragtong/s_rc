import re
with open("src/cascades2_5d/mod.rs", "r") as f:
    text = f.read()

layout_replacement = """
    let layout = BindGroupLayoutDescriptor::new(
        "cascades2_5d_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer::<f32>(false),
                uniform_buffer::<CascadesUniforms>(false),
            ),
        ),
    );
"""
text = re.sub(r"let layout = BindGroupLayoutDescriptor::new\(.*?  \);", layout_replacement.strip(), text, flags=re.DOTALL)
with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)
