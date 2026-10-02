import sys
code = open("src/shade.rs").read()
code = code.replace("let Some(normal_tex) = &prepass.normal else {\n        return;\n    };", """let Some(normal_tex) = &prepass.normal else {
        return;
    };
    let mut uni = UniformBuffer::from(extracted_view.clip_from_world.inverse());
    uni.write_buffer(&render_device, &render_queue);
""")
open("src/shade.rs", "w").write(code)
