//! GDExtension entry point. Coordinate mapping between planet-core (Z-up,
//! +X = lat 0 / lon 0) and Godot (Y-up): `godot = (x, z, -y)`, so the north
//! pole is +Y and lon 0 is +X.

use godot::prelude::*;

mod planet_mesh;

struct PlanetCreatorExtension;

#[gdextension]
unsafe impl ExtensionLibrary for PlanetCreatorExtension {}

/// planet-core direction -> Godot vector (proper rotation, determinant +1).
pub(crate) fn to_godot(d: [f64; 3]) -> Vector3 {
    Vector3::new(d[0] as f32, d[2] as f32, -d[1] as f32)
}
