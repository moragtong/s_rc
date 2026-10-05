#[derive(Component)]
pub struct HeightmapViewMesh;

use crate::heightmap::{DEBUG_LAYER, Heightmap};
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::*;
use bevy::shader::ShaderRef;

pub struct HeightmapViewPlugin;

impl Plugin for HeightmapViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<HeightmapMaterial>::default())
            .add_systems(Startup, setup_view);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct HeightmapMaterial {
    #[texture(0)]
    #[sampler(1)]
    heightmap: Handle<Image>,
    #[uniform(2)]
    channel: f32,
}

impl Material for HeightmapMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/heightmap_view.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/heightmap_view.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

fn setup_view(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HeightmapMaterial>>,
    heightmap_res: Res<Heightmap>,
) {
    // Spawning a highly subdivided plane so we can do vertex displacement!
    let mesh = Plane3d::default()
        .mesh()
        .size(40.0, 32.0)
        .subdivisions(256)
        .build();
    commands.spawn((
        Mesh3d(meshes.add(mesh.clone())),
        MeshMaterial3d(materials.add(HeightmapMaterial {
            heightmap: heightmap_res.texture.clone(),
            channel: 1.0,
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RenderLayers::layer(DEBUG_LAYER),
        HeightmapViewMesh,
    ));
    commands.spawn((
        Mesh3d(meshes.add(mesh.clone())),
        MeshMaterial3d(materials.add(HeightmapMaterial {
            heightmap: heightmap_res.texture.clone(),
            channel: 0.0,
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RenderLayers::layer(DEBUG_LAYER),
        HeightmapViewMesh,
    ));
}
