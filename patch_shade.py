import sys
code = open("src/shade.rs").read()
code = code.replace("Some(BlendState::ADD)", "Some(BlendState { color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add }, alpha: BlendComponent::REPLACE })")
code = code.replace("let (view_target, extracted_view, prepass) = view.single();", "let Some((view_target, extracted_view, prepass)) = view.iter().next() else { return; };")
open("src/shade.rs", "w").write(code)
