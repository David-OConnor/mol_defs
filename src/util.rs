//! Misc.

use std::{
    f64::consts::TAU,
    fmt,
    fmt::{Display, Formatter},
};

use lin_alg::f64::{Quaternion, Vec3};

use crate::molecules::Atom;

pub fn rotate_atoms_about_point(atoms: &mut [Atom], pivot: Vec3, rotator: Quaternion) {
    for a in atoms {
        let rel = a.posit - pivot;
        a.posit = pivot + rotator.rotate_vec(rel);
    }
}
