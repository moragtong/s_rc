//! The analytic 3D test room: the 2D testbed's room scene, lifted to 3D.
//!
//! One definition feeds everything that has to agree on the geometry: the
//! Bevy meshes (raster view), the parry3d SDF bake, and the Mitsuba
//! reference scene (see `export_json`). Everything is axis-aligned
//! boxes, so an exact analytic SDF also exists to validate the bake against.
//!
//! Units are meters, Y up. The room's interior spans x, z in [-4, 4] and y
//! in [0, 3], closed by walls, a ceiling and a ground slab, with one doorway
//! in the -X wall. The only light is a sun (no sky yet), low in the -X sky,
//! so it lights the outside and shines in through the doorway at an angle
//! onto the floor. Everything else inside the room is lit only by light
//! bouncing off that sunlit patch: the test of the cascades' indirect light.

use bevy::prelude::*;

/// Doorway in the -X wall.
const DOOR_HALF_WIDTH: f32 = 0.6;
const DOOR_HEIGHT: f32 = 2.2;
const ROOM_HALF: f32 = 4.0;
const ROOM_HEIGHT: f32 = 3.0;

#[derive(Clone, Copy, Debug)]
pub struct SceneBox {
    pub center: Vec3,
    pub half_size: Vec3,
    pub albedo: Vec3,
}

impl SceneBox {
    /// Exact signed distance, for validating the bake (tests; the clipmap is
    /// validated against Mitsuba by tools/mitsuba_depth.py).
    #[cfg(test)]
    pub fn sdf(&self, p: Vec3) -> f32 {
        let q = (p - self.center).abs() - self.half_size;
        q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
    }
}

/// A sun: a distant disk light.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// Unit vector from the scene toward the sun.
    pub toward: Vec3,
    /// Irradiance on a surface facing the sun (linear RGB).
    pub irradiance: Vec3,
    /// Angular radius of the sun's disk; the real sun's is ~0.27 degrees.
    pub angular_radius: f32,
}

pub struct Scene3d {
    pub boxes: Vec<SceneBox>,
    pub sun: Sun,
}

impl Scene3d {
    /// The room, with walls and ceiling `wall_thickness` thick.
    pub fn room(wall_thickness: f32) -> Self {
        let t = wall_thickness;
        let wall = Vec3::new(0.8, 0.75, 0.7);
        let ground = Vec3::splat(0.5);
        let (h, y_mid) = (ROOM_HEIGHT, ROOM_HEIGHT / 2.0);
        let half_len = ROOM_HALF + t / 2.0;
        let mut boxes = vec![
            // Ground slab, extending well past the room.
            SceneBox { center: Vec3::new(0.0, -0.1, 0.0), half_size: Vec3::new(12.0, 0.1, 12.0), albedo: ground },
            // Ceiling.
            SceneBox { center: Vec3::new(0.0, h + t / 2.0, 0.0), half_size: Vec3::new(half_len, t / 2.0, half_len), albedo: wall },
            // +X, -Z, +Z walls.
            SceneBox { center: Vec3::new(ROOM_HALF, y_mid, 0.0), half_size: Vec3::new(t / 2.0, h / 2.0, half_len), albedo: wall },
            SceneBox { center: Vec3::new(0.0, y_mid, -ROOM_HALF), half_size: Vec3::new(half_len, h / 2.0, t / 2.0), albedo: wall },
            SceneBox { center: Vec3::new(0.0, y_mid, ROOM_HALF), half_size: Vec3::new(half_len, h / 2.0, t / 2.0), albedo: wall },
        ];
        // -X wall, split around the doorway: two side pieces and a lintel.
        let side_half = (half_len - DOOR_HALF_WIDTH) / 2.0;
        for sign in [-1.0, 1.0] {
            boxes.push(SceneBox {
                center: Vec3::new(-ROOM_HALF, y_mid, sign * (DOOR_HALF_WIDTH + side_half)),
                half_size: Vec3::new(t / 2.0, h / 2.0, side_half),
                albedo: wall,
            });
        }
        boxes.push(SceneBox {
            center: Vec3::new(-ROOM_HALF, (DOOR_HEIGHT + h) / 2.0, 0.0),
            half_size: Vec3::new(t / 2.0, (h - DOOR_HEIGHT) / 2.0, DOOR_HALF_WIDTH),
            albedo: wall,
        });

        // Low in the -X sky and a little toward -Z, so it shines in through
        // the doorway at an angle onto the floor.
        let sun = Sun {
            toward: Vec3::new(-1.0, 0.75, -0.35).normalize(),
            irradiance: Vec3::new(3.0, 2.85, 2.6),
            angular_radius: 0.27f32.to_radians(),
        };
        Self { boxes, sun }
    }
}

/// Camera parameters for a reference render: a pinhole at `position`
/// looking along `forward`, vertical field of view, image size.
pub struct ExportCamera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov_y_degrees: f32,
    pub width: u32,
    pub height: u32,
    /// Clipmap tier windows at export time: (origin in voxels, voxel size,
    /// resolution), so the comparison can attribute each hit to a tier.
    pub tiers: Vec<(IVec3, f32, i32)>,
}

impl Scene3d {
    /// JSON for the Mitsuba tools: boxes, the sun and the camera.
    pub fn export_json(&self, camera: &ExportCamera) -> String {
        let v = |v: Vec3| format!("[{}, {}, {}]", v.x, v.y, v.z);
        let boxes: Vec<String> = self
            .boxes
            .iter()
            .map(|b| format!("{{\"center\": {}, \"half_size\": {}, \"albedo\": {}}}", v(b.center), v(b.half_size), v(b.albedo)))
            .collect();
        let sun = format!(
            "{{\"toward\": {}, \"irradiance\": {}, \"angular_radius_degrees\": {}}}",
            v(self.sun.toward),
            v(self.sun.irradiance),
            self.sun.angular_radius.to_degrees()
        );
        let tiers: Vec<String> = camera
            .tiers
            .iter()
            .map(|(o, voxel, res)| format!("{{\"origin\": [{}, {}, {}], \"voxel\": {voxel}, \"resolution\": {res}}}", o.x, o.y, o.z))
            .collect();
        format!(
            "{{\n  \"camera\": {{\"position\": {}, \"forward\": {}, \"up\": {}, \"fov_y_degrees\": {}, \"width\": {}, \"height\": {}}},\n  \"clipmap\": [{}],\n  \"boxes\": [\n    {}\n  ],\n  \"sun\": {sun}\n}}\n",
            v(camera.position),
            v(camera.forward),
            v(camera.up),
            camera.fov_y_degrees,
            camera.width,
            camera.height,
            tiers.join(", "),
            boxes.join(",\n    "),
        )
    }
}

/// Marks the box meshes, which the clipmap bakes into its SDF.
#[derive(Component, Clone, Copy)]
pub struct SceneBoxMesh;

/// Spawns the scene for the raster view: one mesh per box (the clipmap bakes
/// these and takes each one's albedo from its material), and a Bevy
/// directional light for the sun so the raster preview is lit.
pub fn spawn(
    scene: &Scene3d,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for b in &scene.boxes {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(b.half_size * 2.0))),
            MeshMaterial3d(materials.add(Color::linear_rgb(b.albedo.x, b.albedo.y, b.albedo.z))),
            Transform::from_translation(b.center),
            SceneBoxMesh,
        ));
    }
    let color = scene.sun.irradiance / scene.sun.irradiance.max_element();
    commands.spawn((
        DirectionalLight {
            color: Color::linear_rgb(color.x, color.y, color.z),
            illuminance: 20_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-scene.sun.toward, Vec3::Y),
    ));
}
