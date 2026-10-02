cat << 'INNER_EOF' >> src/cascades2_5d/mod.rs

pub fn setup_shading(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shading_layout = BindGroupLayoutDescriptor::new(
        "cascades_shading_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer::<f32>(false),
            ),
        ),
    );
    let shader = asset_server.load("shaders/shade_cascades.wgsl");
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("cascades2_5d_shading".into()),
        layout: vec![shading_layout.clone()],
        shader,
        shader_defs: vec![],
        entry_point: Some("shade".into()),
        ..default()
    });
}
INNER_EOF
cargo check
