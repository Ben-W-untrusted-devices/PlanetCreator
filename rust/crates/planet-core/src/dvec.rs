//! Minimal f64 vector helpers (planet-scale positions need doubles).

pub type DVec3 = [f64; 3];

pub fn add(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: DVec3, s: f64) -> DVec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: DVec3, b: DVec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: DVec3, b: DVec3) -> DVec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length(a: DVec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: DVec3) -> DVec3 {
    let l = length(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
