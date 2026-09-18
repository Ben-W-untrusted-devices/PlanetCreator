//! `PlanetMesh`: six cube-sphere faces as MeshInstance3D children, displaced by a
//! baked height layer and textured with the baked RGB layer. No LOD yet; this is
//! the phase-2 starting point and a visual check that the conventions agree.

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
use planet_core::cubesphere::{face_uv_to_dir, uv_to_pixel, FACE_NAMES};
use planet_core::npy::NpyData;

use crate::to_godot;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

#[derive(GodotClass)]
#[class(base = Node3D)]
pub struct PlanetMesh {
    /// Baked cube directory (`res://` or absolute). Empty = plain sphere.
    #[export]
    cube_dir: GString,
    /// Mesh vertices per face edge.
    #[export]
    mesh_res: i32,
    /// Texture pixels per face edge (the baked layer is strided down to this).
    #[export]
    texture_res: i32,
    /// Sphere radius in scene units.
    #[export]
    radius: f32,
    /// Vertical exaggeration applied to height (1 = true scale, invisible at this size).
    #[export]
    height_exaggeration: f32,
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for PlanetMesh {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            cube_dir: "res://../data/cube/4096".into(),
            mesh_res: 128,
            texture_res: 1024,
            radius: 1.0,
            height_exaggeration: 30.0,
            base,
        }
    }

    fn ready(&mut self) {
        self.rebuild();
    }
}

#[godot_api]
impl PlanetMesh {
    /// Drop existing face children and regenerate them from the current exports.
    #[func]
    pub fn rebuild(&mut self) {
        let children = self.base().get_children();
        for child in children.iter_shared() {
            self.base_mut().remove_child(&child);
            child.free();
        }
        let cube = match self.open_cube() {
            Ok(c) => c,
            Err(e) => {
                godot_warn!("PlanetMesh: {e}; building a plain sphere");
                None
            }
        };
        for (face, name) in FACE_NAMES.iter().enumerate() {
            let mut mi = self.build_face(face, cube.as_ref());
            mi.set_name(*name);
            self.base_mut().add_child(&mi);
        }
    }

    fn open_cube(&self) -> Result<Option<BakedCube>, String> {
        if self.cube_dir.is_empty() {
            return Ok(None);
        }
        let path = ProjectSettings::singleton()
            .globalize_path(&self.cube_dir)
            .to_string();
        BakedCube::open(Path::new(&path))
            .map(Some)
            .map_err(|e| format!("{path}: {e}"))
    }

    fn build_face(&self, face: usize, cube: Option<&BakedCube>) -> Gd<MeshInstance3D> {
        let n = self.mesh_res.max(2) as usize;
        let height = cube.and_then(|c| c.layer(face, "height").ok());
        let (hres, hdata) = match &height {
            Some(a) => (a.shape[0], a.to_f32()),
            None => (1, Vec::new()),
        };

        let mut verts = PackedVector3Array::new();
        let mut uvs = PackedVector2Array::new();
        let mut normals = PackedVector3Array::new();
        for i in 0..=n {
            for j in 0..=n {
                let u = j as f64 / n as f64 * 2.0 - 1.0;
                let v = 1.0 - i as f64 / n as f64 * 2.0;
                let dir = face_uv_to_dir(face, u, v);
                let h = if hdata.is_empty() {
                    0.0
                } else {
                    let (pi, pj) = uv_to_pixel(hres, u, v);
                    f64::from(hdata[pi * hres + pj]).max(0.0) // oceans stay at sea level
                };
                let r = f64::from(self.radius)
                    * (1.0 + f64::from(self.height_exaggeration) * h / EARTH_RADIUS_M);
                let g = to_godot(dir);
                verts.push(g * r as f32);
                normals.push(g);
                uvs.push(Vector2::new(
                    ((u + 1.0) / 2.0) as f32,
                    ((1.0 - v) / 2.0) as f32,
                ));
            }
        }
        let mut indices = PackedInt32Array::new();
        let row = (n + 1) as i32;
        for i in 0..n as i32 {
            for j in 0..n as i32 {
                let a = i * row + j;
                let (b, c, d) = (a + 1, a + row, a + row + 1);
                // Counter-clockwise seen from outside (Godot front faces are CCW).
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

        let mut mat = StandardMaterial3D::new_gd();
        match cube.and_then(|c| self.face_texture(c, face)) {
            Some(tex) => mat.set_texture(TextureParam::ALBEDO, &tex),
            None => mat.set_albedo(Color::from_rgb(0.2, 0.3, 0.6)),
        }
        let mut mi = MeshInstance3D::new_alloc();
        mi.set_mesh(&mesh);
        mi.set_material_override(&mat);
        mi
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
        let img = Image::create_from_data(out as i32, out as i32, false, Format::RGB8, &bytes)?;
        ImageTexture::create_from_image(&img)
    }
}
