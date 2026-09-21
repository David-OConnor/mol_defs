//! Assemble single-stranded DNA and antiparallel duplexes from canonical B-DNA
//! heavy-atom coordinates and Amber residue templates. Templates supply topology,
//! force-field properties and local hydrogen geometry.

use super::{b_dna_coordinates as coordinates, *};

const RISE: f64 = 3.38;
const TWIST: f64 = 36.0_f64.to_radians();

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// An orthonormal, right-handed frame. Using proper rotations preserves the
/// template's sugar stereochemistry and hydrogen names (including H2′/H2″).
fn frame(a: Vec3, b: Vec3) -> io::Result<[Vec3; 3]> {
    if a.magnitude() < 1e-8 || a.cross(b).magnitude() < 1e-8 {
        return Err(invalid("Degenerate DNA template hydrogen frame"));
    }
    let x = a.to_normalized();
    let z = a.cross(b).to_normalized();
    Ok([x, z.cross(x), z])
}

/// Map heavy atoms to the canonical repeat. Transfer each H using a local
/// bonded frame, retaining its template bond length and local orientation.
/// Terminal hydroxyls, amino groups and methyls use a heavy grandparent to
/// define rotation around their single heavy-neighbor bond.
fn positions(template: &TemplateData, nt: Nucleotide) -> io::Result<Vec<Vec3>> {
    let n = template.atoms.len();
    let mut result = vec![Vec3::new_zero(); n];
    let mut adjacent = vec![Vec::new(); n];
    let indices: HashMap<_, _> = template
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i))
        .collect();
    for bond in &template.bonds {
        let a = *indices
            .get(&bond.atom_0_sn)
            .ok_or_else(|| invalid("Invalid DNA template bond"))?;
        let b = *indices
            .get(&bond.atom_1_sn)
            .ok_or_else(|| invalid("Invalid DNA template bond"))?;
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    for (i, atom) in template.atoms.iter().enumerate() {
        if atom.element == Hydrogen {
            continue;
        }
        let name = atom
            .type_in_res_general
            .as_deref()
            .ok_or_else(|| invalid("Unnamed DNA atom"))?;
        let (r, phi, z) = coordinates::cylindrical(nt, name)
            .ok_or_else(|| invalid(format!("No canonical DNA coordinate for {nt}:{name}")))?;
        let phi = phi.to_radians();
        result[i] = Vec3::new(r * phi.cos(), r * phi.sin(), z);
    }
    for (i, atom) in template.atoms.iter().enumerate() {
        if atom.element != Hydrogen {
            continue;
        }
        let heavy = |j: &usize| template.atoms[*j].element != Hydrogen;
        let parent = *adjacent[i]
            .iter()
            .find(|j| heavy(j))
            .ok_or_else(|| invalid("DNA template hydrogen has no heavy parent"))?;
        let neighbors: Vec<_> = adjacent[parent].iter().copied().filter(heavy).collect();
        let a = *neighbors
            .first()
            .ok_or_else(|| invalid("Missing DNA hydrogen frame neighbor"))?;
        let b = if neighbors.len() > 1 {
            neighbors[1]
        } else {
            *adjacent[a]
                .iter()
                .find(|&&j| j != parent && heavy(&j))
                .ok_or_else(|| invalid("Missing DNA hydrogen frame grandparent"))?
        };
        let old = frame(
            template.atoms[a].posit - template.atoms[parent].posit,
            template.atoms[b].posit - template.atoms[parent].posit,
        )?;
        let new = frame(result[a] - result[parent], result[b] - result[parent])?;
        let h = atom.posit - template.atoms[parent].posit;
        result[i] = result[parent]
            + new[0] * h.dot(old[0])
            + new[1] * h.dot(old[1])
            + new[2] * h.dot(old[2]);
    }
    Ok(result)
}

pub(super) fn build(
    seq: &[Nucleotide],
    templates: &HashMap<String, TemplateData>,
    strands: Strands,
) -> io::Result<(Vec<Atom>, Vec<Bond>, Vec<Residue>)> {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut residues = Vec::new();
    // Each chemical template is transformed once, even for long sequences.
    let mut cached = HashMap::new();
    for chain in 0..if strands == Strands::Double { 2 } else { 1 } {
        let mut previous_o3 = None;
        for j in 0..seq.len() {
            let pair = if chain == 0 { j } else { seq.len() - 1 - j };
            let nt = if chain == 0 {
                seq[pair]
            } else {
                seq[pair].complement()
            };
            let suffix = if seq.len() == 1 {
                "N"
            } else if j == 0 {
                "5"
            } else if j + 1 == seq.len() {
                "3"
            } else {
                ""
            };
            let key = format!("D{}{suffix}", nt.to_str_upper());
            let template = templates
                .get(&key)
                .ok_or_else(|| invalid(format!("Unable to find DNA template {key}")))?;
            if !cached.contains_key(&key) {
                cached.insert(key.clone(), positions(template, nt)?);
            }
            let local = &cached[&key];
            let start = atoms.len();
            let res_i = residues.len();
            let mut serials = HashMap::new();
            let mut names = HashMap::new();
            let (sin, cos) = (TWIST * pair as f64).sin_cos();
            for (a, p) in template.atoms.iter().zip(local) {
                let mut atom = Atom::from(a);
                // A 180° rotation about local X relates the two strands while
                // preserving D-deoxyribose chirality.
                let sign = if chain == 0 { -1.0 } else { 1.0 };
                let x = p.x * cos - sign * p.y * sin;
                let y = p.x * sin + sign * p.y * cos;
                let z = sign * p.z + RISE * pair as f64;
                // Rotate the canonical Z-axis helix onto the output Y axis.
                atom.posit = Vec3::new(x, z, -y);
                atom.serial_number = atoms.len() as u32 + 1;
                atom.residue = Some(res_i);
                atom.chain = Some(chain);
                serials.insert(a.serial_number, atoms.len());
                if let Some(name) = a.type_in_res_general.as_deref() {
                    names.insert(name, atoms.len());
                }
                atoms.push(atom);
            }
            let mut add_bond = |a: usize, b: usize, kind| {
                let mut bond =
                    Bond::new_basic(atoms[a].serial_number, atoms[b].serial_number, kind);
                bond.atom_0 = a;
                bond.atom_1 = b;
                bonds.push(bond);
            };
            for bond in &template.bonds {
                add_bond(
                    serials[&bond.atom_0_sn],
                    serials[&bond.atom_1_sn],
                    bond.bond_type,
                );
            }
            if let Some(o3) = previous_o3 {
                let p = *names
                    .get("P")
                    .ok_or_else(|| invalid("Missing DNA phosphate"))?;
                add_bond(o3, p, BondType::Single);
            }
            previous_o3 = Some(*names.get("O3'").ok_or_else(|| invalid("Missing DNA O3′"))?);
            residues.push(Residue {
                serial_number: res_i as u32 + 1,
                res_type: ResidueType::Other(key),
                atom_sns: (start..atoms.len())
                    .map(|i| atoms[i].serial_number)
                    .collect(),
                atoms: (start..atoms.len()).collect(),
                dihedral: None,
                end: if j == 0 {
                    ResidueEnd::NTerminus
                } else if j + 1 == seq.len() {
                    ResidueEnd::CTerminus
                } else {
                    ResidueEnd::Internal
                },
            });
        }
    }
    Ok((atoms, bonds, residues))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn molecule(seq: &[Nucleotide], strands: Strands) -> MoleculeNucleicAcid {
        let (dna, rna) = load_na_templates().unwrap();
        MoleculeNucleicAcid::from_seq(seq, NucleicAcidType::Dna, strands, &dna, &rna).unwrap()
    }

    fn atom(m: &MoleculeNucleicAcid, res: usize, name: &str) -> usize {
        *m.residues[res]
            .atoms
            .iter()
            .find(|&&i| m.common.atoms[i].type_in_res_general.as_deref() == Some(name))
            .unwrap()
    }

    fn angle(a: Vec3, center: Vec3, b: Vec3) -> f64 {
        (a - center)
            .to_normalized()
            .dot((b - center).to_normalized())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    #[test]
    fn every_dinucleotide_has_physical_backbone_geometry() {
        for a in [A, T, C, G] {
            for b in [A, T, C, G] {
                // Exercise 5′, internal and 3′ templates on both strands.
                let m = molecule(&[a, b, a, b], Strands::Double);
                let atoms = &m.common.atoms;
                let mut links = 0;
                for bond in &m.common.bonds {
                    let (a, b) = (&atoms[bond.atom_0], &atoms[bond.atom_1]);
                    let length = (a.posit - b.posit).magnitude();
                    let range = if a.element == Hydrogen || b.element == Hydrogen {
                        0.90..1.15
                    } else {
                        1.15..1.70
                    };
                    assert!(
                        range.contains(&length),
                        "{:?}—{:?}: {length}",
                        a.type_in_res_general,
                        b.type_in_res_general
                    );
                    if a.residue == b.residue {
                        continue;
                    }
                    links += 1;
                    assert_eq!(a.chain, b.chain);
                    assert_eq!(a.type_in_res_general.as_deref(), Some("O3'"));
                    assert_eq!(b.type_in_res_general.as_deref(), Some("P"));
                    assert!((1.58..1.62).contains(&length), "O3′–P: {length}");
                    let res = b.residue.unwrap();
                    let oxygens = [
                        bond.atom_0,
                        atom(&m, res, "OP1"),
                        atom(&m, res, "OP2"),
                        atom(&m, res, "O5'"),
                    ];
                    for i in 0..4 {
                        for j in i + 1..4 {
                            let value =
                                angle(atoms[oxygens[i]].posit, b.posit, atoms[oxygens[j]].posit);
                            assert!((100.0..121.0).contains(&value), "O–P–O: {value}");
                        }
                    }
                    let c3 = atoms[atom(&m, a.residue.unwrap(), "C3'")].posit;
                    let value = angle(c3, a.posit, b.posit);
                    assert!((115.0..125.0).contains(&value), "C3′–O3′–P: {value}");
                }
                assert_eq!(links, 6);
            }
        }
    }

    #[test]
    fn duplex_pairs_and_single_strand_coordinates_agree() {
        let seq = [A, T, C, G, G, T, A, C];
        let ds = molecule(&seq, Strands::Double);
        let ss = molecule(&seq, Strands::Single);
        for (a, b) in ss.common.atoms.iter().zip(&ds.common.atoms) {
            assert!((a.posit - b.posit).magnitude() < 1e-12);
        }
        for (i, nt) in seq.iter().enumerate() {
            let opposite = 2 * seq.len() - 1 - i;
            let pairs = match nt {
                A => vec![("N1", "N3"), ("N6", "O4")],
                T => vec![("N3", "N1"), ("O4", "N6")],
                C => vec![("N3", "N1"), ("N4", "O6"), ("O2", "N2")],
                G => vec![("N1", "N3"), ("O6", "N4"), ("N2", "O2")],
            };
            for (a, b) in pairs {
                let length = (ds.common.atoms[atom(&ds, i, a)].posit
                    - ds.common.atoms[atom(&ds, opposite, b)].posit)
                    .magnitude();
                assert!((2.7..3.15).contains(&length), "{nt} {a}–{b}: {length}");
            }
        }
        // The same sugar atom advances with a positive right-handed screw.
        let a = ds.common.atoms[atom(&ds, 0, "C1'")].posit;
        let b = ds.common.atoms[atom(&ds, 1, "C1'")].posit;
        assert!((b.y - a.y - RISE).abs() < 1e-10);
        assert!(Vec3::new(a.x, 0., a.z).cross(Vec3::new(b.x, 0., b.z)).y > 0.);
    }

    #[test]
    fn topology_termini_and_serials_are_consistent() {
        for seq in [
            vec![],
            vec![A],
            vec![T],
            vec![C],
            vec![G],
            vec![A, C],
            vec![A, T, C, G].repeat(30),
        ] {
            for strands in [Strands::Single, Strands::Double] {
                let m = molecule(&seq, strands);
                let count = if strands == Strands::Single { 1 } else { 2 };
                assert_eq!(m.residues.len(), seq.len() * count);
                for (ri, res) in m.residues.iter().enumerate() {
                    for (&i, &sn) in res.atoms.iter().zip(&res.atom_sns) {
                        let a = &m.common.atoms[i];
                        assert_eq!(a.serial_number, sn);
                        assert_eq!(sn as usize, i + 1);
                        assert_eq!(a.residue, Some(ri));
                        assert_eq!(a.chain, Some(ri / seq.len()));
                        assert!(
                            a.posit.x.is_finite() && a.posit.y.is_finite() && a.posit.z.is_finite()
                        );
                    }
                    let names: Vec<_> = res
                        .atoms
                        .iter()
                        .map(|&i| m.common.atoms[i].type_in_res_general.as_deref().unwrap())
                        .collect();
                    assert_eq!(names.contains(&"HO5'"), ri % seq.len() == 0);
                    assert_eq!(names.contains(&"HO3'"), ri % seq.len() + 1 == seq.len());
                    assert_eq!(names.contains(&"P"), ri % seq.len() != 0);
                }
                for b in &m.common.bonds {
                    assert_eq!(m.common.atoms[b.atom_0].serial_number, b.atom_0_sn);
                    assert_eq!(m.common.atoms[b.atom_1].serial_number, b.atom_1_sn);
                }
                // Exactly one covalent connected component per strand.
                let mut visited = vec![false; m.common.atoms.len()];
                let mut components = 0;
                for i in 0..visited.len() {
                    if visited[i] {
                        continue;
                    }
                    components += 1;
                    let mut stack = vec![i];
                    while let Some(j) = stack.pop() {
                        if visited[j] {
                            continue;
                        }
                        visited[j] = true;
                        stack.extend(&m.common.adjacency_list[j]);
                    }
                }
                assert_eq!(components, if seq.is_empty() { 0 } else { count });
            }
        }
    }

    #[test]
    fn missing_template_returns_an_error() {
        assert!(build(&[A], &HashMap::new(), Strands::Single).is_err());
    }

    #[test]
    fn sugars_keep_template_chirality_and_atoms_do_not_overlap() {
        let m = molecule(&[A, T, C, G].repeat(3), Strands::Double);
        let (templates, _) = load_na_templates().unwrap();
        for (ri, res) in m.residues.iter().enumerate() {
            let ResidueType::Other(key) = &res.res_type else {
                unreachable!()
            };
            let template = &templates[key];
            for (center, neighbors) in [
                ("C1'", ["O4'", "C2'", "H1'"]),
                ("C3'", ["C2'", "C4'", "O3'"]),
                ("C4'", ["O4'", "C3'", "C5'"]),
            ] {
                let before = template.find_atom_by_name(center).unwrap().posit;
                let after = m.common.atoms[atom(&m, ri, center)].posit;
                let old = neighbors.map(|n| template.find_atom_by_name(n).unwrap().posit - before);
                let new = neighbors.map(|n| m.common.atoms[atom(&m, ri, n)].posit - after);
                assert!(
                    old[0].cross(old[1]).dot(old[2]) * new[0].cross(new[1]).dot(new[2]) > 0.,
                    "{key} {center}"
                );
            }
        }
        for (i, a) in m.common.atoms.iter().enumerate() {
            for (j, b) in m.common.atoms.iter().enumerate().skip(i + 1) {
                if m.common.adjacency_list[i].contains(&j) {
                    continue;
                }
                let distance = (a.posit - b.posit).magnitude();
                let min = if a.element == Hydrogen || b.element == Hydrogen {
                    1.2
                } else {
                    2.0
                };
                assert!(
                    distance > min,
                    "Nonbonded {:?}:{:?} and {:?}:{:?}: {distance}",
                    a.residue,
                    a.type_in_res_general,
                    b.residue,
                    b.type_in_res_general
                );
            }
        }
    }
}
