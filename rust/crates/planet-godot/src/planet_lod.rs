//! `PlanetLod`: real-time cube-sphere quadtree terrain with a floating origin.
//!
//! The camera node sits at the engine origin and only rotates; this node keeps the
//! camera's planet-centred position in f64 and places every patch relative to it each
//! frame. Patch meshes are built relative to their own centre, so f32 precision is
//! only ever spent on distances of one patch or the camera-to-patch offset.
//! Level transitions are hidden with skirts.

use std::collections::HashMap;
use std::path::Path;

use godot::classes::base_material_3d::TextureParam;
use godot::classes::image::Format;
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::{
    ArrayMesh, INode3D, Image, ImageTexture, MeshInstance3D, Node3D, ProjectSettings,
    StandardMaterial3D,
};
use godot::prelude::*;
use planet_core::cube::BakedCube;
use planet_core::cubesphere::{dir_to_face_uv, dir_to_latlon, face_uv_to_dir, latlon_to_dir};
use planet_core::dvec::{self, DVec3};
use planet_core::heightfield::{detail_height, FaceHeight};
use planet_core::npy::NpyData;
use planet_core::quadtree::{select, LodParams, QuadNode};

use crate::to_godot;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Godot (Y-up) vector -> planet-core (Z-up) vector; inverse of `to_godot`.
fn from_godot(v: Vector3) -> DVec3 {
    [v.x as f64, -v.z as f64, v.y as f64]
}

#[derive(GodotClass)]
#[class(base = Node3D)]
pub struct PlanetLod {
    /// Baked cube directory (`res://` or absolute).
    #[export]
    cube_dir: GString,
    #[export]
    radius_m: f64,
    /// Deepest quadtree level (18 ~ 1 m cells with patch_res 32).
    #[export]
    max_level: i32,
    /// Split while camera distance < split_factor * node size.
    #[export]
    split_factor: f64,
    /// Quads per patch edge.
    #[export]
    patch_res: i32,
    /// Texture pixels per face edge (the baked rgb is strided down to this).
    #[export]
    texture_res: i32,
    /// Fractal detail amplitude relative to wavelength (0 = baked data only).
    #[export]
    roughness: f64,
    /// Finest detail wavelength in metres.
    #[export]
    fine_m: f64,
    /// New patches built per frame (bounds hitches).
    #[export]
    patches_per_frame: i32,
    #[export]
    start_lat: f64,
    #[export]
    start_lon: f64,
    #[export]
    start_altitude_m: f64,

    cam: DVec3,
    faces: Option<Box<[FaceHeight; 6]>>,
    materials: Vec<Gd<StandardMaterial3D>>,
    patches: HashMap<QuadNode, Gd<MeshInstance3D>>,
    data_res_m: f64,
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for PlanetLod {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            cube_dir: "res://../data/cube/4096".into(),
            radius_m: EARTH_RADIUS_M,
            max_level: 18,
            split_factor: 2.5,
            patch_res: 32,
            texture_res: 4096,
            roughness: 0.06,
            fine_m: 2.0,
            patches_per_frame: 12,
            start_lat: 20.0,
            start_lon: 0.0,
            start_altitude_m: 20_000_000.0,
            cam: [3.0 * EARTH_RADIUS_M, 0.0, 0.0],
            faces: None,
            materials: Vec::new(),
            patches: HashMap::new(),
            data_res_m: 2400.0,
            base,
        }
    }

    fn ready(&mut self) {
        let lat = self.start_lat;
        let lon = self.start_lon;
        let alt = self.start_altitude_m;
        self.load_cube();
        self.set_camera_lat_lon_alt(lat, lon, alt);
    }

    fn process(&mut self, _delta: f64) {
        self.update_lod();
    }
}

#[godot_api]
impl PlanetLod {
    /// Move the camera by an engine-space (Y-up, metres) delta.
    #[func]
    pub fn move_camera(&mut self, delta: Vector3) {
        self.cam = dvec::add(self.cam, from_godot(delta));
    }

    #[func]
    pub fn set_camera_lat_lon_alt(&mut self, lat: f64, lon: f64, alt_m: f64) {
        let dir = latlon_to_dir(lat, lon);
        let ground = self.ground_height(dir);
        self.cam = dvec::scale(dir, self.radius_m + ground + alt_m);
    }

    /// Local "up" (radial) in engine space.
    #[func]
    pub fn get_up(&self) -> Vector3 {
        to_godot(dvec::normalize(self.cam))
    }

    /// Height of the camera above the terrain directly below it, in metres.
    #[func]
    pub fn get_altitude(&self) -> f64 {
        let dir = dvec::normalize(self.cam);
        dvec::length(self.cam) - self.radius_m - self.ground_height(dir)
    }

    /// Terrain height (m above the sphere) under the camera.
    #[func]
    pub fn get_ground_height(&self) -> f64 {
        self.ground_height(dvec::normalize(self.cam))
    }

    #[func]
    pub fn get_lat_lon(&self) -> Vector2 {
        let (lat, lon) = dir_to_latlon(self.cam);
        Vector2::new(lat as f32, lon as f32)
    }

    #[func]
    pub fn get_patch_count(&self) -> i32 {
        self.patches.len() as i32
    }

    #[func]
    pub fn get_camera_position_m(&self) -> Vector3 {
        Vector3::new(self.cam[0] as f32, self.cam[1] as f32, self.cam[2] as f32)
    }

    fn load_cube(&mut self) {
        let path = ProjectSettings::singleton()
            .globalize_path(&self.cube_dir)
            .to_string();
        let cube = match BakedCube::open(Path::new(&path)) {
            Ok(c) => c,
            Err(e) => {
                godot_error!("PlanetLod: cannot open cube {path}: {e}");
                return;
            }
        };
        let n = cube.meta.resolution;
        self.data_res_m = std::f64::consts::FRAC_PI_2 * self.radius_m / n as f64;
        let mut faces: Vec<FaceHeight> = Vec::with_capacity(6);
        for face in 0..6 {
            match cube.layer(face, "height") {
                Ok(a) => faces.push(FaceHeight {
                    n,
                    data: a.to_f32(),
                }),
                Err(e) => {
                    godot_error!("PlanetLod: {e}");
                    return;
                }
            }
            let mut mat = StandardMaterial3D::new_gd();
            mat.set_roughness(1.0);
            match self.face_texture(&cube, face) {
                Some(tex) => mat.set_texture(TextureParam::ALBEDO, &tex),
                None => mat.set_albedo(Color::from_rgb(0.3, 0.3, 0.3)),
            }
            self.materials.push(mat);
        }
        self.faces = Some(Box::new(faces.try_into().ok().expect("six faces")));
    }

    fn face_texture(&self, cube: &BakedCube, face: usize) -> Option<Gd<ImageTexture>> {
        let rgb = cube.layer(face, "rgb").ok()?;
        let NpyData::U8(px) = &rgb.data else {
            return None;
        };
        let src = rgb.shape[0];
        let stride = (src / self.texture_res.max(1) as usize).max(1);
        let out = src / stride;
        let mut bytes = PackedByteArray::new();
        bytes.resize(out * out * 3);
        {
            let dst = bytes.as_mut_slice();
            for i in 0..out {
                for j in 0..out {
                    let s = ((i * stride) * src + j * stride) * 3;
                    let d = (i * out + j) * 3;
                    dst[d..d + 3].copy_from_slice(&px[s..s + 3]);
                }
            }
        }
        let mut img = Image::create_from_data(out as i32, out as i32, false, Format::RGB8, &bytes)?;
        img.generate_mipmaps();
        ImageTexture::create_from_image(&img)
    }

    /// Height above the sphere for a direction, including detail down to `fine_m`.
    fn ground_height(&self, dir: DVec3) -> f64 {
        self.height_at(dir, self.fine_m)
    }

    fn height_at(&self, dir: DVec3, finest_m: f64) -> f64 {
        let Some(faces) = &self.faces else { return 0.0 };
        let (face, u, v) = dir_to_face_uv(dir);
        let base = faces[face].sample(u, v);
        if base <= 0.0 {
            return 0.0;
        }
        let d = if self.roughness > 0.0 {
            detail_height(
                dir,
                self.radius_m,
                self.data_res_m,
                finest_m.max(self.fine_m),
                self.roughness,
                7,
            )
        } else {
            0.0
        };
        let fade = (base / 20.0).clamp(0.0, 1.0);
        (base + d * fade).max(0.5)
    }

    fn update_lod(&mut self) {
        if self.faces.is_none() {
            return;
        }
        let params = LodParams {
            radius: self.radius_m,
            max_level: self.max_level.clamp(0, 30) as u8,
            split_factor: self.split_factor,
        };
        let wanted = select(self.cam, &params);
        let wanted_set: std::collections::HashSet<QuadNode> = wanted.iter().copied().collect();

        // Drop patches no longer wanted.
        let stale: Vec<QuadNode> = self
            .patches
            .keys()
            .filter(|k| !wanted_set.contains(k))
            .copied()
            .collect();
        for k in stale {
            if let Some(mut mi) = self.patches.remove(&k) {
                self.base_mut().remove_child(&mi);
                mi.queue_free();
            }
        }
        // Build a bounded number of new ones, nearest first.
        let mut missing: Vec<QuadNode> = wanted
            .iter()
            .filter(|k| !self.patches.contains_key(k))
            .copied()
            .collect();
        let cam = self.cam;
        let r = self.radius_m;
        missing.sort_by(|a, b| {
            let da = dvec::length(dvec::sub(cam, dvec::scale(a.centre_dir(), r)));
            let db = dvec::length(dvec::sub(cam, dvec::scale(b.centre_dir(), r)));
            da.partial_cmp(&db).unwrap()
        });
        for node in missing
            .into_iter()
            .take(self.patches_per_frame.max(1) as usize)
        {
            let mi = self.build_patch(node);
            self.base_mut().add_child(&mi);
            self.patches.insert(node, mi);
        }
        // Re-place everything relative to the camera.
        let placements: Vec<(QuadNode, Vector3)> = self
            .patches
            .keys()
            .map(|n| (*n, to_godot(dvec::sub(self.patch_origin(*n), self.cam))))
            .collect();
        for (node, rel) in placements {
            if let Some(mi) = self.patches.get_mut(&node) {
                mi.set_position(rel);
            }
        }
    }

    fn patch_origin(&self, node: QuadNode) -> DVec3 {
        let dir = node.centre_dir();
        dvec::scale(
            dir,
            self.radius_m + self.height_at(dir, node.size_m(self.radius_m) / self.patch_res as f64),
        )
    }

    fn build_patch(&self, node: QuadNode) -> Gd<MeshInstance3D> {
        let res = self.patch_res.max(2) as usize;
        let (u0, v0, u1, v1) = node.bounds();
        let cell_m = node.size_m(self.radius_m) / res as f64;
        let origin = self.patch_origin(node);
        let skirt = cell_m * 0.5 + 5.0;
        let g = res + 3; // grid including a skirt ring on each side

        // Surface positions (metres, relative to the patch origin) on a grid one cell wider
        // than the patch on every side, so edge normals see true neighbours. The outer ring
        // then becomes the skirt: same (u, v) as the edge, dropped by `skirt`.
        let mut surf = vec![[0.0f64; 3]; g * g];
        let mut uvs = PackedVector2Array::new();
        let mut dirs = vec![[0.0f64; 3]; g * g];
        for gi in 0..g {
            for gj in 0..g {
                let ii = gi as f64 - 1.0;
                let jj = gj as f64 - 1.0;
                let u = u0 + (u1 - u0) * jj / res as f64;
                let v = v1 - (v1 - v0) * ii / res as f64;
                let dir = face_uv_to_dir(node.face as usize, u, v);
                let h = self.height_at(dir, cell_m);
                surf[gi * g + gj] = dvec::sub(dvec::scale(dir, self.radius_m + h), origin);
                dirs[gi * g + gj] = dir;
                let uc = u.clamp(u0, u1);
                let vc = v.clamp(v0, v1);
                uvs.push(Vector2::new(
                    ((uc + 1.0) * 0.5) as f32,
                    ((1.0 - vc) * 0.5) as f32,
                ));
            }
        }
        let mut verts = PackedVector3Array::new();
        let mut normals = PackedVector3Array::new();
        for gi in 0..g {
            for gj in 0..g {
                let edge = gi == 0 || gj == 0 || gi == g - 1 || gj == g - 1;
                let gi0 = gi.clamp(1, g - 2);
                let gj0 = gj.clamp(1, g - 2);
                let du = dvec::sub(surf[gi0 * g + gj0 + 1], surf[gi0 * g + gj0 - 1]);
                let dv = dvec::sub(surf[(gi0 - 1) * g + gj0], surf[(gi0 + 1) * g + gj0]);
                normals.push(to_godot(dvec::normalize(dvec::cross(du, dv))));
                let p = if edge {
                    // skirt vertex: under the nearest edge vertex, dropped along the radial
                    let e = surf[gi0 * g + gj0];
                    dvec::sub(e, dvec::scale(dirs[gi0 * g + gj0], skirt))
                } else {
                    surf[gi * g + gj]
                };
                verts.push(to_godot(p));
            }
        }
        let mut indices = PackedInt32Array::new();
        let row = g as i32;
        for i in 0..(g - 1) as i32 {
            for j in 0..(g - 1) as i32 {
                let a = i * row + j;
                let (b, c, d) = (a + 1, a + row, a + row + 1);
                indices.extend([a, b, c, b, d, c]);
            }
        }
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        arrays.set(ArrayType::VERTEX.ord() as usize, &verts.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
        arrays.set(ArrayType::TEX_UV.ord() as usize, &uvs.to_variant());
        arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());
        let mut mesh = ArrayMesh::new_gd();
        mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
        let mut mi = MeshInstance3D::new_alloc();
        mi.set_mesh(&mesh);
        mi.set_material_override(&self.materials[node.face as usize]);
        mi.set_name(&format!(
            "f{}l{}x{}y{}",
            node.face, node.level, node.x, node.y
        ));
        mi
    }
}
