//! Misc.

use std::f64::consts::TAU;
use std::fmt;
use std::fmt::{Display, Formatter};
use lin_alg::f64::{Quaternion, Vec3};

use crate::molecules::Atom;

/// Returns the centroid of a set of atoms, and the maximum distance of any of them from the
/// origin along a single axis. The latter is a cheap proxy for molecule size; it avoids computing
/// magnitudes.
pub fn mol_center_size(atoms: &[Atom]) -> (Vec3, f32) {
    let mut sum = Vec3::new_zero();
    let mut max_dim = 0.;

    for atom in atoms {
        sum += atom.posit;

        // Cheaper than calculating magnitude.
        if atom.posit.x.abs() > max_dim {
            max_dim = atom.posit.x.abs();
        }
        if atom.posit.y.abs() > max_dim {
            max_dim = atom.posit.y.abs();
        }
        if atom.posit.z.abs() > max_dim {
            max_dim = atom.posit.z.abs();
        }
    }

    (sum / (atoms.len() as f64), max_dim as f32)
}

pub fn rotate_atoms_about_point(atoms: &mut [Atom], pivot: Vec3, rotator: Quaternion) {
    for a in atoms {
        let rel = a.posit - pivot;
        a.posit = pivot + rotator.rotate_vec(rel);
    }
}

