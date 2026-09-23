// Scene primitives and materials shared by the rc2d scenes.
#define_import_path rc2d::sdf

struct SceneSample {
    dist: f32,
    emission: vec3<f32>,
    // 1 = emission restricted to the spotlight cone (see spot_factor).
    spot: f32,
    // Lambertian reflectance; 0 for lights and black occluders.
    albedo: vec3<f32>,
}

fn sd_circle(p: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn sd_box(p: vec2<f32>, center: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let q = abs(p - center) - half_size;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}

fn closer(a: SceneSample, dist: f32, emission: vec3<f32>) -> SceneSample {
    if dist < a.dist {
        return SceneSample(dist, emission, 0.0, vec3(0.0));
    }
    return a;
}

fn closer_spot(a: SceneSample, dist: f32, emission: vec3<f32>) -> SceneSample {
    if dist < a.dist {
        return SceneSample(dist, emission, 1.0, vec3(0.0));
    }
    return a;
}

fn closer_diffuse(a: SceneSample, dist: f32, albedo: vec3<f32>) -> SceneSample {
    if dist < a.dist {
        return SceneSample(dist, vec3(0.0), 0.0, albedo);
    }
    return a;
}

// Spotlight: axis along +x, 45 degree half-angle, 4 degree soft edge.
const SPOT_AXIS: vec2<f32> = vec2(1.0, 0.0);
const SPOT_COS_OUTER: f32 = 0.6593; // cos(49 deg)
const SPOT_COS_INNER: f32 = 0.7431; // cos(42 deg)

// A ray travelling along `ray_dir` sees light the spot emitted along
// -ray_dir. Applied at the hit, so cascades and reference share it exactly.
fn spot_factor(ray_dir: vec2<f32>) -> f32 {
    return smoothstep(SPOT_COS_OUTER, SPOT_COS_INNER, dot(-ray_dir, SPOT_AXIS));
}
