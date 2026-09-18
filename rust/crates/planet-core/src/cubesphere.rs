//! Cube-sphere parameterisation. See the Python docstring for the convention;
//! in short: right-handed, +Z north pole, +X through (lat 0, lon 0), faces
//! 0..5 = +X -X +Y -Y +Z -Z, tangent-warped (u, v) in [-1, 1]^2.

use std::f64::consts::{FRAC_PI_4, PI};

pub type Vec3 = [f64; 3];

/// (n, tu, tv) per face with `tu x tv == n`.
pub const FACE_AXES: [[Vec3; 3]; 6] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
];

pub const FACE_NAMES: [&str; 6] = ["px", "nx", "py", "ny", "pz", "nz"];

pub fn warp(u: f64) -> f64 {
    (u * FRAC_PI_4).tan()
}

pub fn unwarp(s: f64) -> f64 {
    s.atan() * 4.0 / PI
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn normalize(d: Vec3) -> Vec3 {
    let n = dot(d, d).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

/// Unit direction for face coordinates.
pub fn face_uv_to_dir(face: usize, u: f64, v: f64) -> Vec3 {
    let [n, tu, tv] = FACE_AXES[face];
    let (s, t) = (warp(u), warp(v));
    normalize([
        n[0] + s * tu[0] + t * tv[0],
        n[1] + s * tu[1] + t * tv[1],
        n[2] + s * tu[2] + t * tv[2],
    ])
}

/// Inverse of [`face_uv_to_dir`]; `d` need not be normalised.
pub fn dir_to_face_uv(d: Vec3) -> (usize, f64, f64) {
    let mut axis = 0;
    for k in 1..3 {
        if d[k].abs() > d[axis].abs() {
            axis = k;
        }
    }
    let face = axis * 2 + usize::from(d[axis] < 0.0);
    let [n, tu, tv] = FACE_AXES[face];
    let dn = dot(d, n);
    (face, unwarp(dot(d, tu) / dn), unwarp(dot(d, tv) / dn))
}

/// Degrees; lat in [-90, 90], lon in [-180, 180).
pub fn dir_to_latlon(d: Vec3) -> (f64, f64) {
    let n = dot(d, d).sqrt();
    let lat = (d[2] / n).clamp(-1.0, 1.0).asin().to_degrees();
    let mut lon = d[1].atan2(d[0]).to_degrees();
    if lon >= 180.0 {
        lon -= 360.0;
    }
    (lat, lon)
}

pub fn latlon_to_dir(lat_deg: f64, lon_deg: f64) -> Vec3 {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let c = la.cos();
    [c * lo.cos(), c * lo.sin(), la.sin()]
}

/// (u, v) at the centre of pixel (row i, col j) of an n x n face raster.
/// Row 0 is the +v edge, column 0 the -u edge.
pub fn pixel_uv(n: usize, i: usize, j: usize) -> (f64, f64) {
    let u = (j as f64 + 0.5) / n as f64 * 2.0 - 1.0;
    let v = 1.0 - (i as f64 + 0.5) / n as f64 * 2.0;
    (u, v)
}

/// Nearest pixel (row, col) for face coordinates; clamps at the edges.
pub fn uv_to_pixel(n: usize, u: f64, v: f64) -> (usize, usize) {
    let max = (n - 1) as f64;
    let j = ((u + 1.0) / 2.0 * n as f64 - 0.5).round().clamp(0.0, max) as usize;
    let i = ((1.0 - v) / 2.0 * n as f64 - 0.5).round().clamp(0.0, max) as usize;
    (i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_are_right_handed() {
        for [n, tu, tv] in FACE_AXES {
            let c = [
                tu[1] * tv[2] - tu[2] * tv[1],
                tu[2] * tv[0] - tu[0] * tv[2],
                tu[0] * tv[1] - tu[1] * tv[0],
            ];
            assert_eq!(c, n);
        }
    }

    #[test]
    fn pixel_round_trip() {
        for n in [1usize, 8, 4096] {
            for (i, j) in [(0, 0), (n - 1, n - 1), (n / 2, n / 3)] {
                let (u, v) = pixel_uv(n, i, j);
                assert_eq!(uv_to_pixel(n, u, v), (i, j));
            }
        }
    }
}
