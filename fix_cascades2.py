import re

with open("src/cascades2_5d/mod.rs", "r") as f:
    text = f.read()

# Replace all of update_frame up to its closing brace
pattern = r"fn update_frame\([\s\S]*?^}"

new_update_frame = """fn update_frame(
    mut frame: ResMut<Cascades2_5dFrame>,
    heightmap: Option<Res<Heightmap>>,
    sun: Option<Res<crate::scene3d::Sun>>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
) {
    let (Some(heightmap), Ok(camera)) = (heightmap, camera.get_single()) else { return };
    if sun.is_some_and(|s| s.is_changed()) {
        frame.lighting += 1;
    }
    frame.active = true;
    frame.camera = camera.translation();
    frame.heightmap = heightmap.clone();
    frame.generation = 1; # mock generation for now
}"""

text = re.sub(pattern, new_update_frame, text, flags=re.MULTILINE | re.DOTALL)

# Let's fix run_cascades as well since it requires ClipmapTextures
run_casc_pattern = r"fn run_cascades\([\s\S]*?(?=\n    let \(Some)"
new_run_casc_sig = """fn run_cascades(
    frame: Res<Cascades2_5dFrame>,
    gpu: Option<Res<Cascades2_5dGpu>>,
    sun: Option<Res<crate::scene3d::Sun>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut params: Local<DynamicUniformBuffer<Params>>,
    mut cleared_generation: Local<Option<u32>>,
    mut lighting: Local<Option<u32>>,
    mut frame_count: Local<u32>,
    mut ctx: RenderContext,
) {"""
text = re.sub(run_casc_pattern, new_run_casc_sig, text, flags=re.MULTILINE|re.DOTALL)

# replace `let (Some(gpu), Some(textures), Some(sun))`
text = text.replace("let (Some(gpu), Some(textures), Some(sun)) = (gpu, textures, sun) else { return };",
                    "let (Some(gpu), Some(sun)) = (gpu, sun) else { return };")

# In the bind group setup, we no longer have `textures.sdf`, `textures.color`, etc.
# But we need them in WGSL. In cascades2_5d.wgsl we just replaced clipmap with heightmap.
# So we must bind the Heightmap's view. But in the extract step we didn't extract the image. We just extracted handle. We need a `RenderAsset<Image>` or `RenderAssets<GpuImage>`.

with open("src/cascades2_5d/mod.rs", "w") as f:
    f.write(text)
