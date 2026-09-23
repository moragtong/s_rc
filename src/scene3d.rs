//! The analytic 3D test room: the 2D testbed's room scene, lifted to 3D.
//!
//! One definition feeds everything that has to agree on the geometry: the
//! Bevy meshes (raster view), the parry3d SDF bake, and the Mitsuba
//! reference scene (see `export_json`). Everything is axis-aligned
//! boxes, so an exact analytic SDF also exists to validate the bake against.
//!
//! Units are meters, Y up. The room's interior spans x, z in [-4, 4] and y
//! in [0, 3], closed by walls, a ceiling and a ground slab, with one doorway
//! in the -X wall. A spotlight outside, up and to the side of the doorway,
//! aims at the wall beside it, so only its cone edge enters the room. Three
//! point lights sit outside the other walls.

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

/// A small emissive sphere. `radiance` is its surface radiance (linear RGB).
#[derive(Clone, Copy, Debug)]
pub struct SphereLight {
    pub center: Vec3,
    pub radius: f32,
    pub radiance: Vec3,
    /// Some for the spotlight: emission only within a cone around `dir`.
    pub spot: Option<Spot>,
}

#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub dir: Vec3,
    pub half_angle: f32,
}

pub struct Scene3d {
    pub boxes: Vec<SceneBox>,
    pub lights: Vec<SphereLight>,
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

        let lights = vec![
            // Aimed along +X at the wall beside the doorway; the doorway is
            // ~42 degrees off-axis, just inside the 45 degree cone.
            SphereLight {
                center: Vec3::new(-6.0, 1.6, -1.8),
                radius: 0.15,
                radiance: Vec3::new(140.0, 90.0, 40.0),
                spot: Some(Spot { dir: Vec3::X, half_angle: 45f32.to_radians() }),
            },
            SphereLight { center: Vec3::new(0.0, 2.0, -6.0), radius: 0.15, radiance: Vec3::new(20.0, 40.0, 120.0), spot: None },
            SphereLight { center: Vec3::new(6.0, 1.5, 1.0), radius: 0.15, radiance: Vec3::new(90.0, 20.0, 70.0), spot: None },
            SphereLight { center: Vec3::new(1.0, 1.0, 6.0), radius: 0.15, radiance: Vec3::new(20.0, 90.0, 30.0), spot: None },
        ];
        Self { boxes, lights }
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
    /// JSON for `tools/mitsuba_depth.py`: boxes, lights and the camera.
    pub fn export_json(&self, camera: &ExportCamera) -> String {
        let v = |v: Vec3| format!("[{}, {}, {}]", v.x, v.y, v.z);
        let boxes: Vec<String> = self
            .boxes
            .iter()
            .map(|b| format!("{{\"center\": {}, \"half_size\": {}, \"albedo\": {}}}", v(b.center), v(b.half_size), v(b.albedo)))
            .collect();
        let lights: Vec<String> = self
            .lights
            .iter()
            .map(|l| {
                let spot = l.spot.map_or("null".to_string(), |s| {
                    format!("{{\"dir\": {}, \"half_angle_degrees\": {}}}", v(s.dir), s.half_angle.to_degrees())
                });
                format!(
                    "{{\"center\": {}, \"radius\": {}, \"radiance\": {}, \"spot\": {spot}}}",
                    v(l.center),
                    l.radius,
                    v(l.radiance)
                )
            })
            .collect();
        let tiers: Vec<String> = camera
            .tiers
            .iter()
            .map(|(o, voxel, res)| format!("{{\"origin\": [{}, {}, {}], \"voxel\": {voxel}, \"resolution\": {res}}}", o.x, o.y, o.z))
            .collect();
        format!(
            "{{\n  \"camera\": {{\"position\": {}, \"forward\": {}, \"up\": {}, \"fov_y_degrees\": {}, \"width\": {}, \"height\": {}}},\n  \"clipmap\": [{}],\n  \"boxes\": [\n    {}\n  ],\n  \"lights\": [\n    {}\n  ]\n}}\n",
            v(camera.position),
            v(camera.forward),
            v(camera.up),
            camera.fov_y_degrees,
            camera.width,
            camera.height,
            tiers.join(", "),
            boxes.join(",\n    "),
            lights.join(",\n    ")
        )
    }
}

/// Marks the box meshes, which the clipmap bakes into its SDF.
#[derive(Component, Clone, Copy)]
pub struct SceneBoxMesh;

/// Spawns the scene for the raster view: one mesh per box, and each light
/// as an emissive sphere plus a Bevy light so the raster preview is lit.
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
    for light in &scene.lights {
        let color = Color::linear_rgb(light.radiance.x, light.radiance.y, light.radiance.z);
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(light.radius))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: color.to_linear(),
                ..default()
            })),
            Transform::from_translation(light.center),
        ));
        let normalized = (light.radiance / light.radiance.max_element()).extend(1.0);
        let light_color = Color::linear_rgba(normalized.x, normalized.y, normalized.z, 1.0);
        match light.spot {
            Some(spot) => commands.spawn((
                SpotLight {
                    color: light_color,
                    intensity: 2_000_000.0,
                    range: 30.0,
                    outer_angle: spot.half_angle,
                    inner_angle: spot.half_angle * 0.9,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_translation(light.center).looking_to(spot.dir, Vec3::Y),
            )),
            None => commands.spawn((
                PointLight {
                    color: light_color,
                    intensity: 400_000.0,
                    range: 30.0,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_translation(light.center),
            )),
        };
    }
}
