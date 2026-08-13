#![allow(non_camel_case_types)]

//! Molecule data structures for computational chemistry and drug discovery.
//!
//! Where [`bio_files`] provides format-level types for reading and writing molecular files, this
//! library provides the application-level types those are loaded into: ones carrying inferred
//! properties, topology, and conformational data.
//!
//! These types are shared between [Molchanica](https://github.com/David-OConnor/molchanica) and its
//! ADME inference library; they live here so neither has to depend on the other.
//!
//! The `render` feature adds surface-mesh generation, and the mesh and electron-density fields used
//! to draw molecules. It pulls in a GPU stack, so leave it off for headless use such as ML inference.

/// RGB, each channel in the range 0.0 to 1.0.
pub type Color = (f32, f32, f32);

/// Read a little-endian primitive out of a byte slice. Used by our custom binary formats.
#[macro_export]
macro_rules! parse_le {
    ($bytes:expr, $t:ty, $range:expr) => {{ <$t>::from_le_bytes($bytes[$range].try_into().unwrap()) }};
}

/// Write a primitive into a byte slice as little-endian. Used by our custom binary formats.
#[macro_export]
macro_rules! copy_le {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_le_bytes()) }};
}

pub mod bond_inference;
pub mod mol_components;
pub mod molecules;
pub mod properties;
pub mod reflection;
pub mod screening;
pub mod serialization;
pub mod sfc_mesh;
pub mod smiles;
pub mod tautomers;
pub mod util;
