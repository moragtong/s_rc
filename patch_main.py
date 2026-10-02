import sys
code = open("src/main.rs").read()
code = code.replace("use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};", "use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass, NormalPrepass};")
code = code.replace("Camera3d::default(),\n        Msaa::Off,\n        DepthPrepass,", "Camera3d::default(),\n        Msaa::Off,\n        DepthPrepass,\n        NormalPrepass,")
open("src/main.rs", "w").write(code)
