//! Build 3D geometry for a molecule whose coordinates are flat: e.g. a depiction from a 2D-only
//! SDF, which is what PubChem serves for compounds it has no 3D conformer for.

use std::io;

use bio_files::md_params::ForceFieldParams;
use dynamics::{
    ComputationDevice, HydrogenConstraint, MdConfig, MdOverrides, MdState, MolDynamics, SimBoxInit,
    Solvent, params::FfParamSet,
};
use lin_alg::f64::Vec3;
use na_seq::Element::Hydrogen;

use crate::molecules::{common::MoleculeCommon, geom_assignment::estimate_bond_length};

/// Atoms start out of plane by up to ± this, in Å. A perfectly flat start has no out-of-plane
/// force for the minimizer to follow, so sp3 centers and non-aromatic rings would stay flat.
const Z_JITTER: f64 = 0.3;

/// Steepest descent, from a start far from any minimum. It usually converges well before this.
const MAX_MINIMIZE_ITERS: usize = 10_000;

/// GROMACS' default `emtol` of 10 kJ mol⁻¹ nm⁻¹, in kcal mol⁻¹ Å⁻¹. `MdConfig`'s default is 100×
/// looser: fine for settling a structure before MD, but not when the geometry is the end result.
const FORCE_TOL: f32 = 10. * 0.0239005;

impl MoleculeCommon {
    /// Replace flat coordinates, e.g. a 2D depiction, with 3D ones: scale the depiction to
    /// realistic bond lengths, nudge atoms out of plane, then minimize energy with the Amber force
    /// field. The molecule stays centered where it was.
    ///
    /// This finds a local minimum near the depiction; not necessarily the lowest-energy conformer.
    /// Stereochemistry isn't preserved: we don't read wedge and hash bonds from 2D files, so the
    /// nudges choose each stereocenter's configuration. They're deterministic, so repeat runs agree.
    ///
    /// Uses atoms' FF types and partial charges; MD setup infers them if any are missing.
    pub fn make_3d(
        &mut self,
        param_set: &FfParamSet,
        mol_specific: Option<&ForceFieldParams>,
    ) -> io::Result<()> {
        let n = self.atoms.len();
        if n == 0 {
            return Err(io::Error::other("The molecule has no atoms."));
        }

        let centroid_local = self.centroid_local();
        let scale = depiction_scale(self);

        let posits_init: Vec<Vec3> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let mut p = (a.posit - centroid_local) * scale;
                p.z += Z_JITTER * jitter(i);
                p
            })
            .collect();

        let radius = posits_init.iter().map(|p| p.magnitude()).fold(0., f64::max);

        let cfg = MdConfig {
            solvent: Solvent::None,
            max_init_relaxation_iters: None,
            hydrogen_constraint: HydrogenConstraint::Flexible,
            energy_minimization_tolerance: FORCE_TOL,
            // Room for the molecule to fill out, without periodic images entering the cutoff.
            sim_box: SimBoxInit::new_cube((2. * (radius + 16.)) as f32),
            overrides: MdOverrides {
                skip_water_relaxation: true,
                skip_counterion_insertion: true,
                long_range_recip_disabled: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let input = MolDynamics {
            atoms: self.atoms.iter().map(|a| a.to_generic()).collect(),
            bonds: self.bonds.iter().map(|b| b.to_generic()).collect(),
            atom_posits: Some(posits_init),
            adjacency_list: Some(self.adjacency_list.clone()),
            mol_specific_params: mol_specific.cloned(),
            ..Default::default()
        };

        // A single small molecule: GPU setup and transfers would cost more than they save.
        let dev = ComputationDevice::Cpu;
        let (mut md, _) = MdState::new(&dev, &cfg, &[input], param_set).map_err(|e| {
            io::Error::other(format!("Unable to set up energy minimization: {e:?}"))
        })?;

        if md.atoms.len() != n {
            return Err(io::Error::other(
                "Energy minimization changed the atom count.",
            ));
        }

        md.minimize_energy(&dev, MAX_MINIMIZE_ITERS, None);

        let posits: Vec<Vec3> = md.atoms.iter().map(|a| a.posit.into()).collect();
        if posits
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Err(io::Error::other("Energy minimization diverged."));
        }

        // Keep the molecule centered where it was: both internally, and as posed in the scene.
        let centroid_new = posits.iter().fold(Vec3::new_zero(), |acc, p| acc + *p) / n as f64;
        let centroid_posed = self.centroid();

        self.atom_posits = Vec::with_capacity(n);
        for (atom, p) in self.atoms.iter_mut().zip(posits) {
            let rel = p - centroid_new;
            atom.posit = centroid_local + rel;
            self.atom_posits.push(centroid_posed + rel);
        }

        self.is_2d = false;
        Ok(())
    }
}

/// The factor that brings a depiction's bond lengths (often 1, in arbitrary units) to realistic
/// ones. The median over bonds, so a few oddly-drawn ones don't skew it. Only bonds between heavy
/// atoms if there are any: depictions draw bonds to H at inconsistent lengths, and
/// `estimate_bond_length` overestimates those.
fn depiction_scale(mol: &MoleculeCommon) -> f64 {
    let find_ratios = |heavy_only: bool| -> Vec<f64> {
        mol.bonds
            .iter()
            .filter_map(|b| {
                let (a0, a1) = (&mol.atoms[b.atom_0], &mol.atoms[b.atom_1]);
                if heavy_only && (a0.element == Hydrogen || a1.element == Hydrogen) {
                    return None;
                }

                let len = (a0.posit - a1.posit).magnitude();
                (len > 1e-3)
                    .then(|| estimate_bond_length(a0.element, a1.element, b.bond_type) / len)
            })
            .collect()
    };

    let mut ratios = find_ratios(true);
    if ratios.is_empty() {
        ratios = find_ratios(false);
    }
    if ratios.is_empty() {
        return 1.;
    }

    ratios.sort_by(f64::total_cmp);
    ratios[ratios.len() / 2]
}

/// A deterministic stand-in for a random number in [-1, 1), so repeat runs give the same
/// conformer. This is SplitMix64's output mix, applied to the atom index.
fn jitter(i: usize) -> f64 {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;

    (x >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bio_files::Sdf;
    use na_seq::Element;

    use super::*;
    use crate::molecules::small::MoleculeSmall;

    /// Cyclohexanol (PubChem CID 7966), as PubChem's 2D record. A non-aromatic ring and sp3 centers
    /// are what stay wrong if the result is flat.
    const CYCLOHEXANOL_2D: &str = "7966
  -OEChem-09212612342D

 19 19  0     0  0  0  0  0  0999 V2000
    2.8660    1.3450    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660    0.3450    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -0.1550    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321   -0.1550    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -1.1550    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321   -1.1550    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660   -1.6550    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.4030    0.6550    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.7880    0.4276    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3894   -0.2627    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.3426   -0.2627    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.9441    0.4276    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3894   -1.0473    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.7880   -1.7376    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.9441   -1.7376    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.3426   -1.0473    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4675   -2.1300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.2646   -2.1300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.4030    1.6550    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1 19  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  8  1  0  0  0  0
  3  5  1  0  0  0  0
  3  9  1  0  0  0  0
  3 10  1  0  0  0  0
  4  6  1  0  0  0  0
  4 11  1  0  0  0  0
  4 12  1  0  0  0  0
  5  7  1  0  0  0  0
  5 13  1  0  0  0  0
  5 14  1  0  0  0  0
  6  7  1  0  0  0  0
  6 15  1  0  0  0  0
  6 16  1  0  0  0  0
  7 17  1  0  0  0  0
  7 18  1  0  0  0  0
M  END
$$$$
";

    fn angle_deg(a: Vec3, center: Vec3, b: Vec3) -> f64 {
        (a - center)
            .to_normalized()
            .dot((b - center).to_normalized())
            .clamp(-1., 1.)
            .acos()
            .to_degrees()
    }

    fn dihedral_deg(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f64 {
        let (b0, b1, b2) = (p1 - p0, p2 - p1, p3 - p2);
        let (n0, n1) = (b0.cross(b1), b1.cross(b2));
        let m = n0.cross(b1.to_normalized());

        m.dot(n1).atan2(n0.dot(n1)).to_degrees()
    }

    #[test]
    fn make_3d_from_2d_sdf() {
        let param_set = FfParamSet::new_amber().unwrap();
        let mut mol: MoleculeSmall = Sdf::new(CYCLOHEXANOL_2D).unwrap().try_into().unwrap();

        // As on load: FF types, partial charges, and the molecule-specific params from them.
        let mut specific = HashMap::new();
        mol.update_ff_related(&mut specific, param_set.small_mol.as_ref().unwrap(), false);
        let specific = specific.get(&mol.common.ident);

        let common = &mut mol.common;

        assert!(common.posits_are_2d());
        common.is_2d = true;

        let centroid_before = common.centroid();
        common.make_3d(&param_set, specific).unwrap();

        assert!(!common.is_2d);
        assert!(!common.posits_are_2d());
        assert!((common.centroid() - centroid_before).magnitude() < 1e-6);

        let p = &common.atom_posits;
        for (i, atom) in common.atoms.iter().enumerate() {
            assert!((atom.posit - p[i]).magnitude() < 1e-9);
        }

        // Realistic bond lengths.
        for b in &common.bonds {
            let (e0, e1) = (
                common.atoms[b.atom_0].element,
                common.atoms[b.atom_1].element,
            );
            let len = (p[b.atom_0] - p[b.atom_1]).magnitude();

            let expected = match (e0, e1) {
                (Element::Carbon, Element::Carbon) => 1.53,
                (Element::Carbon, Element::Oxygen) | (Element::Oxygen, Element::Carbon) => 1.43,
                (Element::Carbon, Element::Hydrogen) | (Element::Hydrogen, Element::Carbon) => 1.09,
                (Element::Oxygen, Element::Hydrogen) | (Element::Hydrogen, Element::Oxygen) => 0.97,
                _ => unreachable!(),
            };
            assert!(
                (len - expected).abs() < 0.04,
                "{e0:?}-{e1:?} bond length {len:.3} Å; expected about {expected:.2}"
            );
        }

        // Every carbon is sp3; flat, some of its angles would be 90° or 180°.
        for (i, atom) in common.atoms.iter().enumerate() {
            if atom.element != Element::Carbon {
                continue;
            }
            let nbrs = &common.adjacency_list[i];
            assert_eq!(nbrs.len(), 4);

            for (j, &a) in nbrs.iter().enumerate() {
                for &b in &nbrs[j + 1..] {
                    let angle = angle_deg(p[a], p[i], p[b]);
                    assert!(
                        (100. ..=120.).contains(&angle),
                        "Angle at C {i}: {angle:.1}°"
                    );
                }
            }
        }

        // A puckered ring, e.g. a chair's ring torsions are about ±55°. Flat, they'd be 0°.
        let ring = [1, 2, 4, 6, 5, 3]; // Atom indices, in order around the ring.
        for k in 0..6 {
            let [a, b, c, d] = std::array::from_fn(|j| p[ring[(k + j) % 6]]);
            let torsion = dihedral_deg(a, b, c, d);
            assert!(torsion.abs() > 30., "Ring torsion {k}: {torsion:.1}°");
        }
    }
}
