import sys
code = open("src/shade.rs").read()
code = code.replace("uniform_buffer::<CascadesUniforms>(false),", "uniform_buffer::<CascadesUniforms>(false),\n                    uniform_buffer::<Mat4>(false),")

code = code.replace("let (view_target, extracted_view, prepass) = view.into_inner();\n    let Some(prepass) = prepass else { return; };", """let (view_target, extracted_view, prepass) = view.into_inner();
    let Some(prepass) = prepass else { return; };
    let mut uni = UniformBuffer::from(extracted_view.clip_from_world.inverse());
    uni.write_buffer(&render_device, &mut ctx.command_encoder());
""")

code = code.replace("render_data.uniform_buffer.as_entire_binding(),", """render_data.uniform_buffer.as_entire_binding(),
            uni.binding().unwrap(),""")

open("src/shade.rs", "w").write(code)
