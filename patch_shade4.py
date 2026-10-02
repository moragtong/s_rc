import sys
code = open("src/shade.rs").read()
code = code.replace("render_device: Res<RenderDevice>,", "render_device: Res<RenderDevice>,\n    render_queue: Res<RenderQueue>,")
code = code.replace("extracted_view.clip_from_world.inverse()", "extracted_view.world_from_clip")
open("src/shade.rs", "w").write(code)
