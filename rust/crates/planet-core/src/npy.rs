//! Minimal reader for NumPy `.npy` files (C-order, little-endian f32/f64/u8/bool).
//! Enough to load a baked cube without pulling in a full ndarray stack.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an npy file")]
    BadMagic,
    #[error("unsupported npy: {0}")]
    Unsupported(String),
    #[error("malformed header: {0}")]
    Header(String),
}

#[derive(Debug, Clone)]
pub enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values as f32 regardless of storage type (bool -> 0/1).
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            NpyData::F32(v) => v.clone(),
            NpyData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| f32::from(u8::from(x))).collect(),
        }
    }
}

fn header_field<'a>(header: &'a str, key: &str) -> Result<&'a str, NpyError> {
    let k = format!("'{key}':");
    let start = header
        .find(&k)
        .ok_or_else(|| NpyError::Header(format!("missing {key}")))?
        + k.len();
    Ok(header[start..].trim_start())
}

pub fn read_npy(path: &Path) -> Result<NpyArray, NpyError> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(NpyError::BadMagic);
    }
    let header_len = match magic[6] {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            u32::from_le_bytes(b) as usize
        }
        v => return Err(NpyError::Unsupported(format!("version {v}"))),
    };
    let mut hb = vec![0u8; header_len];
    r.read_exact(&mut hb)?;
    let header = String::from_utf8_lossy(&hb).into_owned();

    let descr = header_field(&header, "descr")?;
    let descr = descr[1..].split('\'').next().unwrap_or("");
    if header_field(&header, "fortran_order")?.starts_with("True") {
        return Err(NpyError::Unsupported("fortran_order".into()));
    }
    let shape_s = header_field(&header, "shape")?;
    let close = shape_s
        .find(')')
        .ok_or_else(|| NpyError::Header("shape".into()))?;
    let shape: Vec<usize> = shape_s[1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| NpyError::Header(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let n: usize = shape.iter().product();

    let mut raw = Vec::new();
    r.read_to_end(&mut raw)?;
    let need = |bytes: usize| -> Result<(), NpyError> {
        if raw.len() < n * bytes {
            Err(NpyError::Header(format!(
                "expected {} bytes, got {}",
                n * bytes,
                raw.len()
            )))
        } else {
            Ok(())
        }
    };
    let data = match descr {
        "<f4" => {
            need(4)?;
            NpyData::F32(
                raw.chunks_exact(4)
                    .take(n)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        "<f8" => {
            need(8)?;
            NpyData::F64(
                raw.chunks_exact(8)
                    .take(n)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
        }
        "|u1" => {
            need(1)?;
            raw.truncate(n);
            NpyData::U8(raw)
        }
        "|b1" => {
            need(1)?;
            NpyData::Bool(raw.iter().take(n).map(|&b| b != 0).collect())
        }
        d => return Err(NpyError::Unsupported(format!("dtype {d}"))),
    };
    Ok(NpyArray { shape, data })
}
