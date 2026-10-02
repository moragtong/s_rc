import sys
code = open("src/shade.rs").read()
code = code.replace("uniform_buffer::<Mat4>(false),", "uniform_buffer::<ShadeUniforms>(false),")
code = code.replace("let mut uni = UniformBuffer::from(extracted_view.world_from_clip);", """let mut uni = UniformBuffer::from(ShadeUniforms {
        clip_from_view: extracted_view.clip_from_view,
        world_from_view: extracted_view.world_from_view.compute_matrix(),
    });""")
code = code.replace("struct ShadePipeline", """#[derive(ShaderType, Clone)]
struct ShadeUniforms {
    clip_from_view: Mat4,
    world_from_view: Mat4,
}

#[derive(Resource)]
struct ShadePipeline""")
open("src/shade.rs", "w").write(code)
