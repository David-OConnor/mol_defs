//! For constructing peptides from amino acid sequences. Amber ff19SB residue templates
//! (`amino19.lib`, with `aminont12.lib` and `aminoct12.lib` for the termini) supply atom names,
//! connectivity, force-field types, partial charges and coordinates.
//!
//! # Construction
//!
//! `MoleculePeptide::from_seq` builds an all-atom, single-chain peptide from a sequence supplied
//! in N→C order. The first residue uses its N-terminal (NH3+) template, the last its C-terminal
//! (COO-) template, and the others their internal templates. A single residue is a zwitterion:
//! its C-terminal template, with the N-terminal template's NH3+ hydrogens. Histidine is
//! δ-protonated (HID), matching how we assign His parameters for MD.
//!
//! The backbone is a β-strand (φ = -120°, ψ = 130°), so the chain lies along a straight line.
//! Proline's φ is set by its ring. Consecutive residues are joined by trans peptide bonds with
//! Amber's (parm19) equilibrium geometry. Each template is placed rigidly onto its N, CA and C,
//! so bond lengths, angles and hydrogen positions are those of the template. The exceptions are
//! the amide H and carbonyl O, which are placed in the peptide planes using the template's
//! bond lengths and angles.
//!
//! The templates' own backbones are fully extended (φ = ψ = 180°), and their side-chain rotamers
//! clash with neighboring residues in a chain. We choose rotations about χ1 and χ2 that avoid
//! this; see `resolve_side_chain_clashes`.
//!
//! The result is an unfolded starting structure. Construction is deterministic, and performs no
//! folding or energy minimization.

use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
    io,
    str::FromStr,
};

use bio_files::{
    AtomGeneric, BondGeneric, BondType, ChainGeneric, ResidueEnd, ResidueGeneric, ResidueType,
    create_bonds,
    mol_templates::{TemplateData, load_templates},
};
use dynamics::params::{AMINO_19, AMINO_CT12, AMINO_NT12};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{AminoAcid, AtomTypeInRes, Element};

use crate::molecules::{init_bonds_chains_res, peptide::MoleculePeptide};

// Amber parm19 equilibrium geometry for the peptide bond, which joins templates.
const BOND_C_N: f64 = 1.335;
const ANGLE_CA_C_N: f64 = 116.6_f64.to_radians();
const ANGLE_C_N_CA: f64 = 121.9_f64.to_radians();
// Trans peptide bonds.
const OMEGA: f64 = PI;

// Rotations about χ1 and χ2 we try, relative to the template's rotamer. Multiples of 120° keep
// sp3 substituents staggered. Planar groups, e.g. rings, flip or turn perpendicular.
const ROTS_SP3: [f64; 3] = [0., 2. * PI / 3., -2. * PI / 3.];
const ROTS_PLANAR: [f64; 4] = [0., PI, PI / 2., -PI / 2.];
// Residues farther apart than this along an unfolded chain can't touch.
const ROTAMER_NEIGHBOR_RES: usize = 2;
const ROTAMER_PASSES: usize = 2;

// Backbone torsions of the unfolded chain: a β-strand. Of the extended conformations, this
// one's side chains clash the least with their neighbors.
const PHI: f64 = (-120.0_f64).to_radians();
const PSI: f64 = 130.0_f64.to_radians();

/// Amber residue templates for building peptides, keyed by Amber residue name.
/// E.g. "ALA" (internal), "NALA" (N-terminus), and "CALA" (C-terminus).
#[derive(Clone, Debug, Default)]
pub struct AminoAcidTemplates {
    pub internal: HashMap<String, TemplateData>,
    pub n_terminus: HashMap<String, TemplateData>,
    pub c_terminus: HashMap<String, TemplateData>,
}

/// Loads templates from Amber data built into the binary.
pub fn load_aa_templates() -> io::Result<AminoAcidTemplates> {
    Ok(AminoAcidTemplates {
        internal: load_templates(AMINO_19)?,
        n_terminus: load_templates(AMINO_NT12)?,
        c_terminus: load_templates(AMINO_CT12)?,
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// The Amber residue name of an amino acid's internal template.
fn amber_name(aa: AminoAcid) -> io::Result<&'static str> {
    use AminoAcid::*;

    Ok(match aa {
        Arg => "ARG",
        // Plain HIS is absent from the Amber libraries; see the module docs.
        His => "HID",
        Lys => "LYS",
        Asp => "ASP",
        Glu => "GLU",
        Ser => "SER",
        Thr => "THR",
        Asn => "ASN",
        Gln => "GLN",
        Cys => "CYS",
        Gly => "GLY",
        Pro => "PRO",
        Ala => "ALA",
        Val => "VAL",
        Ile => "ILE",
        Leu => "LEU",
        Met => "MET",
        Phe => "PHE",
        Tyr => "TYR",
        Trp => "TRP",
        Sec => {
            return Err(invalid(
                "There is no Amber template for selenocysteine (Sec)",
            ));
        }
    })
}

fn get_template<'a>(
    templates: &'a HashMap<String, TemplateData>,
    name: &str,
) -> io::Result<&'a TemplateData> {
    templates
        .get(name)
        .ok_or_else(|| invalid(format!("Unable to find the amino acid template {name}")))
}

fn atom_name(atom: &AtomGeneric) -> io::Result<&str> {
    atom.type_in_res_general
        .as_deref()
        .ok_or_else(|| invalid("Unnamed amino acid template atom"))
}

fn posit(template: &TemplateData, name: &str) -> io::Result<Vec3> {
    template
        .find_atom_by_name(name)
        .map(|a| a.posit)
        .ok_or_else(|| invalid(format!("Amino acid template is missing atom {name}")))
}

fn new_bond(bond_type: BondType, atom_0_sn: u32, atom_1_sn: u32) -> BondGeneric {
    BondGeneric {
        bond_type,
        atom_0_sn,
        atom_1_sn,
    }
}

/// An orthonormal, right-handed frame. Using proper rotations preserves the
/// template's chirality.
fn frame(a: Vec3, b: Vec3) -> io::Result<[Vec3; 3]> {
    if a.magnitude() < 1e-8 || a.cross(b).magnitude() < 1e-8 {
        return Err(invalid("Degenerate amino acid template backbone"));
    }
    let x = a.to_normalized();
    let z = a.cross(b).to_normalized();
    Ok([x, z.cross(x), z])
}

/// The angle a–b–c, in radians.
fn bond_angle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    (a - b)
        .to_normalized()
        .dot((c - b).to_normalized())
        .clamp(-1., 1.)
        .acos()
}

/// Place an atom bonded to `c`, from its bond length, the angle b–c–new, and the
/// dihedral angle a–b–c–new. (NeRF)
fn place(a: Vec3, b: Vec3, c: Vec3, bond_len: f64, angle: f64, dihedral: f64) -> Vec3 {
    let bc = (c - b).to_normalized();
    let n = (b - a).cross(bc).to_normalized();
    let m = n.cross(bc);

    c + bc * (-bond_len * angle.cos())
        + m * (bond_len * angle.sin() * dihedral.cos())
        + n * (bond_len * angle.sin() * dihedral.sin())
}

/// The dihedral angle a–b–c–d, in radians, in (-π, π].
fn dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let axis = (c - b).to_normalized();
    let v = (a - b) - axis * (a - b).dot(axis);
    let w = (d - c) - axis * (d - c).dot(axis);

    axis.cross(v).dot(w).atan2(v.dot(w))
}

/// φ for a residue. Proline's ring fixes it: the previous residue's C is opposite CD across N.
fn phi(template: &TemplateData) -> io::Result<f64> {
    if template.find_atom_by_name("H").is_some() {
        return Ok(PHI);
    }

    Ok(dihedral(
        posit(template, "CD")?,
        posit(template, "N")?,
        posit(template, "CA")?,
        posit(template, "C")?,
    ) + PI)
}

/// A single residue: the C-terminal template, with the N-terminal template's NH3+ hydrogens
/// in place of the amide H.
fn zwitterion(aa: AminoAcid, templates: &AminoAcidTemplates) -> io::Result<TemplateData> {
    let name = amber_name(aa)?;
    let mut result = get_template(&templates.c_terminus, &format!("C{name}"))?.clone();
    let n_term = get_template(&templates.n_terminus, &format!("N{name}"))?;

    // Remove the amide H, and its bond.
    if let Some(h) = result.find_atom_by_name("H") {
        let sn = h.serial_number;
        result.atoms.retain(|a| a.serial_number != sn);
        result
            .bonds
            .retain(|b| b.atom_0_sn != sn && b.atom_1_sn != sn);
    }

    // Map the N-terminal template's hydrogens on N onto this template's backbone.
    let old = frame(
        posit(n_term, "CA")? - posit(n_term, "N")?,
        posit(n_term, "C")? - posit(n_term, "N")?,
    )?;
    let n = posit(&result, "N")?;
    let new = frame(posit(&result, "CA")? - n, posit(&result, "C")? - n)?;

    let n_sn = result.find_atom_by_name("N").unwrap().serial_number;
    let n_sn_n_term = n_term.find_atom_by_name("N").unwrap().serial_number;
    let mut next_sn = result
        .atoms
        .iter()
        .map(|a| a.serial_number)
        .max()
        .unwrap_or(0)
        + 1;

    for bond in &n_term.bonds {
        let h_sn = if bond.atom_0_sn == n_sn_n_term {
            bond.atom_1_sn
        } else if bond.atom_1_sn == n_sn_n_term {
            bond.atom_0_sn
        } else {
            continue;
        };

        let h = n_term
            .atoms
            .iter()
            .find(|a| a.serial_number == h_sn)
            .ok_or_else(|| invalid("Invalid amino acid template bond"))?;

        if h.element != Element::Hydrogen {
            continue;
        }

        let d = h.posit - posit(n_term, "N")?;
        result.atoms.push(AtomGeneric {
            serial_number: next_sn,
            posit: n + new[0] * d.dot(old[0]) + new[1] * d.dot(old[1]) + new[2] * d.dot(old[2]),
            ..h.clone()
        });
        result.bonds.push(new_bond(BondType::Single, n_sn, next_sn));

        next_sn += 1;
    }

    Ok(result)
}

/// Atom indices reachable from `start` without crossing `blocked`.
fn downstream(adj: &[Vec<usize>], start: usize, blocked: usize) -> Vec<usize> {
    let mut result = vec![start];
    let mut stack = vec![start];

    while let Some(i) = stack.pop() {
        for &j in &adj[i] {
            if j != blocked && !result.contains(&j) {
                result.push(j);
                stack.push(j);
            }
        }
    }
    result
}

/// Atom indices within 3 bonds of `start`; their distances follow from covalent geometry.
fn within_3_bonds(adj: &[Vec<usize>], start: usize) -> HashSet<usize> {
    let mut result = HashSet::from([start]);
    let mut frontier = vec![start];

    for _ in 0..3 {
        let mut next = Vec::new();
        for &i in &frontier {
            for &j in &adj[i] {
                if result.insert(j) {
                    next.push(j);
                }
            }
        }
        frontier = next;
    }
    result
}

/// A soft penalty for two atoms being closer than their contact distance.
fn clash_penalty(a: &AtomGeneric, b: &AtomGeneric) -> f64 {
    let min = match (
        a.element == Element::Hydrogen,
        b.element == Element::Hydrogen,
    ) {
        (true, true) => 2.0,
        (false, false) => 3.0,
        _ => 2.4,
    };

    let dist = (a.posit - b.posit).magnitude();
    if dist < min { (min - dist).powi(2) } else { 0. }
}

fn rotate_about(atoms: &mut [AtomGeneric], indices: &[usize], pivot: Vec3, axis: Vec3, angle: f64) {
    let rotation = Quaternion::from_axis_angle(axis.to_normalized(), angle);
    for &i in indices {
        atoms[i].posit = pivot + rotation.rotate_vec(atoms[i].posit - pivot);
    }
}

/// The templates' side-chain rotamers often clash with neighboring residues. Choose rotations about
/// χ1 and χ2 that minimize clashes with nearby residues. This is greedy, in sequence order.
/// Atom serial numbers must be indices + 1.
fn resolve_side_chain_clashes(
    atoms: &mut [AtomGeneric],
    bonds: &[BondGeneric],
    residues: &[ResidueGeneric],
) {
    let mut adj = vec![Vec::new(); atoms.len()];
    for bond in bonds {
        let (i, j) = (bond.atom_0_sn as usize - 1, bond.atom_1_sn as usize - 1);
        adj[i].push(j);
        adj[j].push(i);
    }

    let res_atoms: Vec<Vec<usize>> = residues
        .iter()
        .map(|r| r.atom_sns.iter().map(|sn| *sn as usize - 1).collect())
        .collect();

    for _ in 0..ROTAMER_PASSES {
        for (r, indices) in res_atoms.iter().enumerate() {
            let find = |name: &str| {
                indices
                    .iter()
                    .copied()
                    .find(|&i| atoms[i].type_in_res_general.as_deref() == Some(name))
            };

            let (Some(n), Some(ca), Some(cb)) = (find("N"), find("CA"), find("CB")) else {
                continue; // e.g. glycine
            };

            let moving_1 = downstream(&adj, cb, ca);
            // Proline's ring closes on the backbone.
            if moving_1.contains(&n) {
                continue;
            }

            // χ2 rotates about CB and the first atom of a side chain continuing past it.
            let heavy = |i: usize| atoms[i].element != Element::Hydrogen;
            let chi2 = adj[cb]
                .iter()
                .copied()
                .find(|&x| x != ca && heavy(x) && adj[x].iter().any(|&y| y != cb && heavy(y)));

            let (moving_2, rots_2): (Vec<usize>, &[f64]) = match chi2 {
                Some(x) if adj[x].len() == 4 => (downstream(&adj, x, cb), &ROTS_SP3),
                Some(x) => (downstream(&adj, x, cb), &ROTS_PLANAR),
                None => (Vec::new(), &[0.]),
            };

            let res_range = r.saturating_sub(ROTAMER_NEIGHBOR_RES)
                ..(r + ROTAMER_NEIGHBOR_RES + 1).min(res_atoms.len());

            let others: Vec<usize> = res_atoms[res_range]
                .iter()
                .flatten()
                .copied()
                .filter(|i| !moving_1.contains(i))
                .collect();

            let near: Vec<HashSet<usize>> =
                moving_1.iter().map(|&i| within_3_bonds(&adj, i)).collect();

            let score = |atoms: &[AtomGeneric]| {
                let mut result = 0.;
                for (k, &i) in moving_1.iter().enumerate() {
                    // Within the side chain, count each pair once.
                    for &j in others.iter().chain(&moving_1[k + 1..]) {
                        if !near[k].contains(&j) {
                            result += clash_penalty(&atoms[i], &atoms[j]);
                        }
                    }
                }
                result
            };

            let original: Vec<Vec3> = moving_1.iter().map(|&i| atoms[i].posit).collect();

            let apply = |atoms: &mut [AtomGeneric], rot_1: f64, rot_2: f64| {
                for (&i, &posit) in moving_1.iter().zip(&original) {
                    atoms[i].posit = posit;
                }

                if let Some(x) = chi2 {
                    let pivot = atoms[x].posit;
                    let axis = pivot - atoms[cb].posit;
                    rotate_about(atoms, &moving_2, pivot, axis, rot_2);
                }

                let pivot = atoms[cb].posit;
                let axis = pivot - atoms[ca].posit;
                rotate_about(atoms, &moving_1, pivot, axis, rot_1);
            };

            // Prefer the template's rotamer on ties.
            let mut best = (score(atoms), 0., 0.);

            for &rot_1 in &ROTS_SP3 {
                for &rot_2 in rots_2 {
                    apply(atoms, rot_1, rot_2);

                    let candidate = score(atoms);
                    if candidate < best.0 - 1e-9 {
                        best = (candidate, rot_1, rot_2);
                    }
                }
            }

            apply(atoms, best.1, best.2);
        }
    }
}

/// Build atoms, bonds, residues and a chain from templates. Atom serial numbers start at 1,
/// in order.
fn build(
    seq: &[AminoAcid],
    templates: &AminoAcidTemplates,
) -> io::Result<(
    Vec<AtomGeneric>,
    Vec<BondGeneric>,
    Vec<ResidueGeneric>,
    ChainGeneric,
)> {
    let mut atoms: Vec<AtomGeneric> = Vec::new();
    let mut bonds = Vec::new();
    let mut residues = Vec::new();

    // The previous residue's backbone N, CA and C positions, and its C serial number.
    let mut prev: Option<(Vec3, Vec3, Vec3, u32)> = None;

    for (i, &aa) in seq.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i + 1 == seq.len();

        let zwitterion_template;
        let template = if is_first && is_last {
            zwitterion_template = zwitterion(aa, templates)?;
            &zwitterion_template
        } else if is_first {
            get_template(&templates.n_terminus, &format!("N{}", amber_name(aa)?))?
        } else if is_last {
            get_template(&templates.c_terminus, &format!("C{}", amber_name(aa)?))?
        } else {
            get_template(&templates.internal, amber_name(aa)?)?
        };

        let t_n = posit(template, "N")?;
        let t_ca = posit(template, "CA")?;
        let t_c = posit(template, "C")?;

        // The first residue stays in its template's frame.
        let (n, ca, c) = match prev {
            Some((n_prev, ca_prev, c_prev, _)) => {
                let n = place(n_prev, ca_prev, c_prev, BOND_C_N, ANGLE_CA_C_N, PSI);

                let ca = place(
                    ca_prev,
                    c_prev,
                    n,
                    (t_ca - t_n).magnitude(),
                    ANGLE_C_N_CA,
                    OMEGA,
                );

                let c = place(
                    c_prev,
                    n,
                    ca,
                    (t_c - t_ca).magnitude(),
                    bond_angle(t_n, t_ca, t_c),
                    phi(template)?,
                );

                (n, ca, c)
            }
            None => (t_n, t_ca, t_c),
        };

        // Rigidly move the template onto this backbone.
        let old = frame(t_ca - t_n, t_c - t_n)?;
        let new = frame(ca - n, c - n)?;

        let mut sns = HashMap::new();
        let mut atom_sns = Vec::with_capacity(template.atoms.len());

        for atom in &template.atoms {
            let sn = atoms.len() as u32 + 1;
            let name = atom_name(atom)?;
            let d = atom.posit - t_n;

            let mut posit =
                n + new[0] * d.dot(old[0]) + new[1] * d.dot(old[1]) + new[2] * d.dot(old[2]);

            // The amide H and carbonyl O depend on φ and ψ. Place them in the peptide planes,
            // opposite the previous C and next N, keeping the template's bond lengths and angles.
            if name == "H" && !is_first {
                let dihedral = phi(template)? + PI;
                posit = place(
                    c,
                    ca,
                    n,
                    d.magnitude(),
                    bond_angle(t_ca, t_n, atom.posit),
                    dihedral,
                );
            } else if name == "O" && !is_last {
                posit = place(
                    n,
                    ca,
                    c,
                    (atom.posit - t_c).magnitude(),
                    bond_angle(t_ca, t_c, atom.posit),
                    PSI + PI,
                );
            }

            atoms.push(AtomGeneric {
                serial_number: sn,
                posit,
                type_in_res: Some(AtomTypeInRes::from_str(name)?),
                ..atom.clone()
            });

            sns.insert(atom.serial_number, sn);
            atom_sns.push(sn);
        }

        for bond in &template.bonds {
            let (Some(&sn_0), Some(&sn_1)) = (sns.get(&bond.atom_0_sn), sns.get(&bond.atom_1_sn))
            else {
                return Err(invalid("Invalid amino acid template bond"));
            };
            bonds.push(new_bond(bond.bond_type, sn_0, sn_1));
        }

        let sn_of = |name: &str| -> io::Result<u32> {
            template
                .find_atom_by_name(name)
                .map(|a| sns[&a.serial_number])
                .ok_or_else(|| invalid(format!("Amino acid template is missing atom {name}")))
        };

        // The peptide bond.
        if let Some((.., c_prev_sn)) = prev {
            bonds.push(new_bond(BondType::Single, c_prev_sn, sn_of("N")?));
        }

        residues.push(ResidueGeneric {
            serial_number: i as u32 + 1,
            res_type: ResidueType::AminoAcid(aa),
            atom_sns,
            end: if is_first {
                ResidueEnd::NTerminus
            } else if is_last {
                ResidueEnd::CTerminus
            } else {
                ResidueEnd::Internal
            },
        });

        if !is_last {
            prev = Some((n, ca, c, sn_of("C")?));
        }
    }

    resolve_side_chain_clashes(&mut atoms, &bonds, &residues);

    // The templates' bonds are all single. Infer orders from bond length, as we do for proteins
    // loaded from files, keeping the templates' connectivity.
    let orders: HashMap<_, _> = create_bonds(&atoms)
        .into_iter()
        .map(|b| {
            (
                (b.atom_0_sn.min(b.atom_1_sn), b.atom_0_sn.max(b.atom_1_sn)),
                b.bond_type,
            )
        })
        .collect();

    for bond in &mut bonds {
        let key = (
            bond.atom_0_sn.min(bond.atom_1_sn),
            bond.atom_0_sn.max(bond.atom_1_sn),
        );
        if let Some(&order) = orders.get(&key) {
            bond.bond_type = order;
        }
    }

    let chain = ChainGeneric {
        id: "A".to_string(),
        residue_sns: residues.iter().map(|r| r.serial_number).collect(),
        atom_sns: atoms.iter().map(|a| a.serial_number).collect(),
    };

    Ok((atoms, bonds, residues, chain))
}

impl MoleculePeptide {
    /// Build an all-atom, unfolded peptide from an amino acid sequence, in N→C order. See the
    /// `amino_acid` module docs for details.
    pub fn from_seq(seq: &[AminoAcid], templates: &AminoAcidTemplates) -> io::Result<Self> {
        if seq.is_empty() {
            return Err(invalid("The amino acid sequence is empty"));
        }

        let (atoms, bonds, residues, chain) = build(seq, templates)?;

        let (atoms, bonds, residues, chains) =
            init_bonds_chains_res(&atoms, &bonds, &residues, &[chain], &[])?;

        Ok(Self::new(
            format!("Peptide {}aa", seq.len()),
            atoms,
            bonds,
            chains,
            residues,
            HashMap::new(),
            None,
        ))
    }
}
