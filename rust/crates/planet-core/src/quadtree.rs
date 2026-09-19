//! Cube-sphere quadtree: which patches to draw for a given camera position.
//!
//! Each face is a quadtree over (u, v) in [-1, 1]^2. A node at `level` covers a
//! `1 / 2^level` fraction of the face edge. Nodes split while the camera is closer
//! than `split_factor` times the node's size on the ground, so the on-screen
//! triangle size stays roughly constant. Level boundaries are hidden with skirts,
//! so neighbours may differ by any number of levels.

use crate::cubesphere::face_uv_to_dir;
use crate::dvec::{length, scale, sub, DVec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuadNode {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl QuadNode {
    pub fn root(face: u8) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// (u0, v0, u1, v1) in face coordinates. `y` counts from the +v edge like raster rows.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let n = (1u64 << self.level) as f64;
        let w = 2.0 / n;
        let u0 = -1.0 + self.x as f64 * w;
        let v1 = 1.0 - self.y as f64 * w;
        (u0, v1 - w, u0 + w, v1)
    }

    pub fn children(&self) -> [QuadNode; 4] {
        let (f, l, x, y) = (self.face, self.level + 1, self.x * 2, self.y * 2);
        [
            QuadNode {
                face: f,
                level: l,
                x,
                y,
            },
            QuadNode {
                face: f,
                level: l,
                x: x + 1,
                y,
            },
            QuadNode {
                face: f,
                level: l,
                x,
                y: y + 1,
            },
            QuadNode {
                face: f,
                level: l,
                x: x + 1,
                y: y + 1,
            },
        ]
    }

    pub fn centre_uv(&self) -> (f64, f64) {
        let (u0, v0, u1, v1) = self.bounds();
        ((u0 + u1) * 0.5, (v0 + v1) * 0.5)
    }

    /// Unit direction through the node centre.
    pub fn centre_dir(&self) -> DVec3 {
        let (u, v) = self.centre_uv();
        face_uv_to_dir(self.face as usize, u, v)
    }

    /// Approximate ground extent of the node in metres (quarter circumference / 2^level).
    pub fn size_m(&self, radius: f64) -> f64 {
        std::f64::consts::FRAC_PI_2 * radius / (1u64 << self.level) as f64
    }
}

pub struct LodParams {
    pub radius: f64,
    pub max_level: u8,
    /// Split while camera distance < split_factor * node size.
    pub split_factor: f64,
}

/// Leaf nodes to render for a camera at `cam` (planet-centred metres).
/// Nodes whose centre faces away from the camera beyond the horizon are still returned
/// at coarse level; culling is left to the renderer.
pub fn select(cam: DVec3, p: &LodParams) -> Vec<QuadNode> {
    let mut out = Vec::new();
    for face in 0..6u8 {
        visit(QuadNode::root(face), cam, p, &mut out);
    }
    out
}

fn visit(node: QuadNode, cam: DVec3, p: &LodParams, out: &mut Vec<QuadNode>) {
    let centre = scale(node.centre_dir(), p.radius);
    let dist = length(sub(cam, centre));
    let size = node.size_m(p.radius);
    if node.level < p.max_level && dist < p.split_factor * size {
        for c in node.children() {
            visit(c, cam, p, out);
        }
    } else {
        out.push(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_tile_the_face() {
        let n = QuadNode::root(0);
        assert_eq!(n.bounds(), (-1.0, -1.0, 1.0, 1.0));
        let c = n.children();
        assert_eq!(c[0].bounds(), (-1.0, 0.0, 0.0, 1.0)); // top-left: +v edge
        assert_eq!(c[3].bounds(), (0.0, -1.0, 1.0, 0.0));
        let mut area = 0.0;
        for k in c {
            let (u0, v0, u1, v1) = k.bounds();
            area += (u1 - u0) * (v1 - v0);
        }
        assert!((area - 4.0).abs() < 1e-12);
    }

    #[test]
    fn far_camera_gets_six_roots_and_close_camera_refines() {
        let p = LodParams {
            radius: 6.371e6,
            max_level: 12,
            split_factor: 2.0,
        };
        let far = select([0.0, 0.0, 1.0e9], &p);
        assert_eq!(far.len(), 6);
        let near = select([6.371e6 + 1000.0, 0.0, 0.0], &p);
        assert!(near.len() > 6);
        assert!(near.iter().any(|n| n.level == 12 && n.face == 0));
        assert!(near.iter().filter(|n| n.face == 1).all(|n| n.level <= 2)); // far side stays coarse
    }
}
