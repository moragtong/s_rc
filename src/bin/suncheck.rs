use bevy::math::{Quat, Vec3};
use bevy::transform::components::Transform;
fn main() {
    for az in [0f32, 90., 270.] {
        let elevation = 30f32.to_radians();
        let azr = az.to_radians();
        let mut t = Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y);
        t.rotation = Quat::from_rotation_y(azr) * Quat::from_rotation_x(-elevation);
        let fwd = t.forward().as_vec3();
        println!("az={} forward={:+.4} sun_toward={:+.4}", az, fwd, -fwd);
    }
}
