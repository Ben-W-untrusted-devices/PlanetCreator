//! Height and colour lookups for a face: baked data below its resolution, fractal
//! detail above it. The Python model owns everything coarser than ~2.4 km; this is the
//! "procedural + assets" band of docs/tech-demo.md.

use crate::cubesphere::{dir_to_face_uv, uv_to_pixel};
use crate::dvec::DVec3;

/// Baked height raster for one face (row 0 = +v edge).
pub struct FaceHeight {
    pub n: usize,
    pub data: Vec<f32>,
}

impl FaceHeight {
    /// Bilinear sample at face coordinates; clamps at the edges.
    pub fn sample(&self, u: f64, v: f64) -> f64 {
        let n = self.n as f64;
        let x = ((u + 1.0) * 0.5 * n - 0.5).clamp(0.0, n - 1.0);
        let y = ((1.0 - v) * 0.5 * n - 0.5).clamp(0.0, n - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.n - 1), (y0 + 1).min(self.n - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let at = |i: usize, j: usize| self.data[i * self.n + j] as f64;
        let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
        let bot = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
        top * (1.0 - fy) + bot * fy
    }

    pub fn nearest(&self, u: f64, v: f64) -> f32 {
        let (i, j) = uv_to_pixel(self.n, u, v);
        self.data[i * self.n + j]
    }
}

/// Hash-based 3D value noise, smooth, in [-1, 1]. Cheap and dependency-free; good
/// enough for sub-data terrain detail (replace with gradient noise for close-ups).
fn hash3(x: i64, y: i64, z: i64, seed: u32) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (z as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ (seed as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
    h ^= h >> 31;
    h = h.wrapping_mul(0x7FB5_D329_728E_A185);
    h ^= h >> 27;
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

pub fn value_noise(p: DVec3, seed: u32) -> f64 {
    let f = [p[0].floor(), p[1].floor(), p[2].floor()];
    let t = [
        smooth(p[0] - f[0]),
        smooth(p[1] - f[1]),
        smooth(p[2] - f[2]),
    ];
    let (x, y, z) = (f[0] as i64, f[1] as i64, f[2] as i64);
    let mut acc = 0.0;
    for (dz, wz) in [(0, 1.0 - t[2]), (1, t[2])] {
        for (dy, wy) in [(0, 1.0 - t[1]), (1, t[1])] {
            for (dx, wx) in [(0, 1.0 - t[0]), (1, t[0])] {
                acc += hash3(x + dx, y + dy, z + dz, seed) * wx * wy * wz;
            }
        }
    }
    acc
}

/// Fractal detail displacement (metres) for a unit direction, octaves between
/// `coarse_m` (the data resolution, where amplitude starts) and `fine_m`.
/// Amplitude per octave is proportional to the octave's wavelength (a rough
/// self-affine landscape) scaled by `roughness`.
pub fn detail_height(
    dir: DVec3,
    radius: f64,
    coarse_m: f64,
    fine_m: f64,
    roughness: f64,
    seed: u32,
) -> f64 {
    let mut wavelength = coarse_m;
    let mut out = 0.0;
    let mut octave = 0u32;
    while wavelength > fine_m && octave < 16 {
        let freq = radius / wavelength; // cycles across the unit sphere
        let p = [dir[0] * freq, dir[1] * freq, dir[2] * freq];
        out += value_noise(p, seed.wrapping_add(octave)) * wavelength * roughness;
        wavelength *= 0.5;
        octave += 1;
    }
    out
}

/// Full height (metres above the sphere) for a direction: baked + detail (land only;
/// the ocean surface stays at 0).
pub fn surface_height(
    faces: &[FaceHeight; 6],
    dir: DVec3,
    radius: f64,
    data_res_m: f64,
    fine_m: f64,
    roughness: f64,
) -> f64 {
    let (face, u, v) = dir_to_face_uv(dir);
    let base = faces[face].sample(u, v);
    if base <= 0.0 {
        return 0.0;
    }
    let d = detail_height(dir, radius, data_res_m, fine_m, roughness, 7);
    // fade detail out near the coast so beaches do not get holes
    let fade = (base / 20.0).clamp(0.0, 1.0);
    (base + d * fade).max(0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_sampling_is_exact_at_pixel_centres_and_smooth_between() {
        let n = 4;
        let data: Vec<f32> = (0..16).map(|k| k as f32).collect();
        let f = FaceHeight { n, data };
        let (u, v) = crate::cubesphere::pixel_uv(n, 1, 2);
        assert!((f.sample(u, v) - 6.0).abs() < 1e-9);
        let (u2, v2) = crate::cubesphere::pixel_uv(n, 1, 3);
        let mid = f.sample((u + u2) * 0.5, v2);
        assert!((mid - 6.5).abs() < 1e-9);
    }

    #[test]
    fn value_noise_is_bounded_and_continuous() {
        let a = value_noise([1.3, 2.7, 0.1], 1);
        let b = value_noise([1.3001, 2.7, 0.1], 1);
        assert!(a.abs() <= 1.0 && (a - b).abs() < 1e-2);
    }

    #[test]
    fn detail_amplitude_scales_with_roughness() {
        let d = [0.3, 0.5, 0.8];
        let a = detail_height(d, 6.371e6, 2400.0, 10.0, 0.1, 7);
        let b = detail_height(d, 6.371e6, 2400.0, 10.0, 0.2, 7);
        assert!((b - 2.0 * a).abs() < 1e-9);
    }
}
