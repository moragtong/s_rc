sed -i '/shading_layout =/c\    let shading_layout = BindGroupLayoutDescriptor::new(\n        "cascades_shading_layout",\n        \&BindGroupLayoutEntries::sequential(\n            ShaderStages::COMPUTE,\n            (texture_2d(TextureSampleType::Float { filterable: false }),),\n        ),\n    );' src/cascades2_5d/mod.rs
cargo check
