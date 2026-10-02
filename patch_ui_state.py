import sys
code = open("src/ui.rs").read()
code = code.replace("#[derive(Resource)]\npub struct LightingState {", "#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]\npub struct LightingState {")
code = code.replace("app.init_resource::<LightingState>()", "app.init_resource::<LightingState>()\n            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<LightingState>::default())")
open("src/ui.rs", "w").write(code)
