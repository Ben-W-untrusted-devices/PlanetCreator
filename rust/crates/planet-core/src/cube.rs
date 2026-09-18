//! A baked cube on disk: `root/meta.json` + `root/<face>/<layer>.npy`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cubesphere::FACE_NAMES;
use crate::npy::{read_npy, NpyArray, NpyError};

#[derive(Debug, Deserialize)]
pub struct LayerInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CubeMeta {
    pub resolution: usize,
    pub faces: Vec<String>,
    pub layers: HashMap<String, LayerInfo>,
}

#[derive(Debug, thiserror::Error)]
pub enum CubeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("meta.json: {0}")]
    Meta(#[from] serde_json::Error),
    #[error("layer {0}: {1}")]
    Layer(String, NpyError),
    #[error("cube was baked with a different face order")]
    FaceOrder,
}

pub struct BakedCube {
    pub root: PathBuf,
    pub meta: CubeMeta,
}

impl BakedCube {
    pub fn open(root: &Path) -> Result<Self, CubeError> {
        let meta: CubeMeta =
            serde_json::from_str(&std::fs::read_to_string(root.join("meta.json"))?)?;
        if meta.faces != FACE_NAMES {
            return Err(CubeError::FaceOrder);
        }
        Ok(Self {
            root: root.to_path_buf(),
            meta,
        })
    }

    pub fn layer(&self, face: usize, name: &str) -> Result<NpyArray, CubeError> {
        let path = self.root.join(FACE_NAMES[face]).join(format!("{name}.npy"));
        read_npy(&path).map_err(|e| CubeError::Layer(name.to_string(), e))
    }
}
