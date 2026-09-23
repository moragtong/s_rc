// Room scene for rc2d: thin diffuse walls, one entrance, a spotlight
// outside it and more lights around the outside.
#define_import_path rc2d::room

#import rc2d::sdf::{SceneSample, sd_box, sd_circle, closer, closer_spot, closer_diffuse}

const WALL_ALBEDO: vec3<f32> = vec3(0.8, 0.75, 0.7);

// A 280x280 room (wall centers x 180..460, y 140..420) with 2px diffuse
// walls, thinner than the probe spacing at s0 = 2, and one 28px entrance in
// the left wall at y 262..290. The spotlight sits outside, up and left of
// the entrance, aimed along +x at the wall above it: only the lower edge of
// its cone reaches through the entrance, as a diagonal beam. Three more
// lights sit outside the other walls and light only their outer faces, so
// any of their color on an inner face is light that crossed a wall. Keep in
// sync with src/bin/rc2d/room.rs.
fn room_scene(p: vec2<f32>, light_pos: vec2<f32>) -> SceneSample {
    var s = SceneSample(1e9, vec3(0.0), 0.0, vec3(0.0));
    s = closer_spot(s, sd_circle(p, light_pos, 10.0), vec3(14.0, 9.0, 4.0));
    s = closer(s, sd_circle(p, vec2(330.0, 70.0), 9.0), vec3(2.0, 4.0, 12.0));
    s = closer(s, sd_circle(p, vec2(490.0, 200.0), 7.0), vec3(9.0, 2.0, 7.0));
    s = closer(s, sd_circle(p, vec2(400.0, 470.0), 8.0), vec3(2.0, 9.0, 3.0));
    s = closer_diffuse(s, sd_box(p, vec2(320.0, 140.0), vec2(141.0, 1.0)), WALL_ALBEDO);
    s = closer_diffuse(s, sd_box(p, vec2(320.0, 420.0), vec2(141.0, 1.0)), WALL_ALBEDO);
    s = closer_diffuse(s, sd_box(p, vec2(460.0, 280.0), vec2(1.0, 141.0)), WALL_ALBEDO);
    // Left wall, split by the entrance.
    s = closer_diffuse(s, sd_box(p, vec2(180.0, 200.5), vec2(1.0, 61.5)), WALL_ALBEDO);
    s = closer_diffuse(s, sd_box(p, vec2(180.0, 355.5), vec2(1.0, 65.5)), WALL_ALBEDO);
    return s;
}
