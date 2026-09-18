//! Cross-language check: values exported by scripts/export_cubesphere_vectors.py.

use planet_core::cubesphere::*;
use planet_core::npy::{read_npy, NpyData};
use serde::Deserialize;
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

#[derive(Deserialize)]
struct Dir {
    dir: [f64; 3],
    face: usize,
    u: f64,
    v: f64,
    lat: f64,
    lon: f64,
}

#[derive(Deserialize)]
struct Pixel {
    face: usize,
    n: usize,
    i: usize,
    j: usize,
    u: f64,
    v: f64,
    lat: f64,
    lon: f64,
}

#[derive(Deserialize)]
struct Vectors {
    directions: Vec<Dir>,
    pixels: Vec<Pixel>,
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn matches_python_directions() {
    let v: Vectors =
        serde_json::from_str(&std::fs::read_to_string(data("cubesphere_vectors.json")).unwrap())
            .unwrap();
    assert!(!v.directions.is_empty());
    for d in &v.directions {
        let (face, u, vv) = dir_to_face_uv(d.dir);
        assert_eq!(face, d.face);
        assert!(
            close(u, d.u) && close(vv, d.v),
            "uv mismatch for {:?}",
            d.dir
        );
        let back = face_uv_to_dir(face, u, vv);
        assert!(back.iter().zip(&d.dir).all(|(a, b)| close(*a, *b)));
        let (lat, lon) = dir_to_latlon(d.dir);
        assert!(close(lat, d.lat) && close(lon, d.lon));
        let d2 = latlon_to_dir(lat, lon);
        assert!(d2.iter().zip(&d.dir).all(|(a, b)| close(*a, *b)));
    }
}

#[test]
fn matches_python_pixels() {
    let v: Vectors =
        serde_json::from_str(&std::fs::read_to_string(data("cubesphere_vectors.json")).unwrap())
            .unwrap();
    for p in &v.pixels {
        let (u, vv) = pixel_uv(p.n, p.i, p.j);
        assert!(close(u, p.u) && close(vv, p.v));
        let (lat, lon) = dir_to_latlon(face_uv_to_dir(p.face, u, vv));
        assert!(
            close(lat, p.lat) && close(lon, p.lon),
            "face {} pixel ({}, {})",
            p.face,
            p.i,
            p.j
        );
    }
}

#[test]
fn reads_numpy_fixtures() {
    let a = read_npy(&data("f32_3x4.npy")).unwrap();
    assert_eq!(a.shape, vec![3, 4]);
    match &a.data {
        NpyData::F32(v) => assert_eq!(v[5], 2.5),
        _ => panic!("dtype"),
    }
    let b = read_npy(&data("u8_2x2x3.npy")).unwrap();
    assert_eq!(b.shape, vec![2, 2, 3]);
    assert!(matches!(&b.data, NpyData::U8(v) if v[11] == 11));
    let c = read_npy(&data("bool_2x3.npy")).unwrap();
    assert!(matches!(&c.data, NpyData::Bool(v) if v == &[true, false, true, false, false, true]));
    assert_eq!(c.to_f32()[2], 1.0);
}
