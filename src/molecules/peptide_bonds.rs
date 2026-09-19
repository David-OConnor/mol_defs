//! Apply mmCIF component topology to the loaded atom instances.

use std::{collections::HashMap, io};

use bio_files::BondType;

use crate::{
    bond_inference::create_hydrogen_bonds_single_mol,
    mmcif_edit::CifDoc,
    molecules::{Bond, peptide::MoleculePeptide},
};

impl MoleculePeptide {
    /// Retain the original mmCIF and supplement inferred bonds with its explicit component
    /// topology. Call after loading atoms: `MmCif` itself does not retain `_chem_comp_bond`.
    pub fn set_source_cif(&mut self, text: String) -> io::Result<()> {
        let doc = CifDoc::new(&text)?;
        self.apply_component_bonds(&doc);
        self.source_cif = Some(text);
        Ok(())
    }

    pub(super) fn apply_component_bonds(&mut self, doc: &CifDoc) {
        let (Some(components), Some(sites)) =
            (doc.category("_chem_comp_bond"), doc.category("_atom_site"))
        else {
            return;
        };

        let mut definitions = HashMap::<&str, Vec<(&str, &str, BondType)>>::new();
        for row in 0..components.len() {
            let (Some(comp), Some(a), Some(b), Some(order)) = (
                components.get(row, "comp_id"),
                components.get(row, "atom_id_1"),
                components.get(row, "atom_id_2"),
                components.get(row, "value_order"),
            ) else {
                continue;
            };
            let Ok(order) = order.to_ascii_lowercase().parse() else {
                continue;
            };
            definitions.entry(comp).or_default().push((a, b, order));
        }

        let by_serial: HashMap<_, _> = self
            .common
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| (atom.serial_number, i))
            .collect();

        // Resolve names within each actual instance, never across copies of a component.
        // Nonpolymers often have label_seq_id '.', so retain author sequence/insertion IDs.
        let mut instances = HashMap::<[&str; 6], HashMap<&str, Vec<usize>>>::new();
        for row in 0..sites.len() {
            let get = |tag| sites.get(row, tag).unwrap_or(".");
            let comp = get("label_comp_id");
            if !definitions.contains_key(comp) {
                continue;
            }
            let Ok(sn) = get("id").parse::<u32>() else {
                continue;
            };
            let Some(&i) = by_serial.get(&sn) else {
                // E.g. an alternate conformer removed by the loader.
                continue;
            };
            let key = [
                comp,
                get("label_asym_id"),
                get("label_seq_id"),
                get("auth_seq_id"),
                get("pdbx_PDB_ins_code"),
                get("pdbx_PDB_model_num"),
            ];
            instances
                .entry(key)
                .or_default()
                .entry(get("label_atom_id"))
                .or_default()
                .push(i);
        }

        let pair = |a: usize, b: usize| (a.min(b), a.max(b));
        let mut existing: HashMap<_, _> = self
            .common
            .bonds
            .iter()
            .enumerate()
            .map(|(i, bond)| (pair(bond.atom_0, bond.atom_1), i))
            .collect();

        for (key, names) in instances {
            for &(a, b, bond_type) in &definitions[key[0]] {
                let (Some(left), Some(right)) = (names.get(a), names.get(b)) else {
                    // Dictionary hydrogens and unobserved atoms need not have coordinates.
                    continue;
                };
                for &i in left {
                    for &j in right {
                        let atom_0 = &self.common.atoms[i];
                        let atom_1 = &self.common.atoms[j];
                        if i == j {
                            continue;
                        }
                        if let (Some(a), Some(b)) =
                            (&atom_0.alt_conformation_id, &atom_1.alt_conformation_id)
                            && a != b
                        {
                            continue;
                        }

                        if let Some(&index) = existing.get(&pair(i, j)) {
                            // Explicit bond order takes precedence over a distance guess.
                            self.common.bonds[index].bond_type = bond_type;
                        } else {
                            existing.insert(pair(i, j), self.common.bonds.len());
                            self.common.bonds.push(Bond {
                                bond_type,
                                atom_0_sn: atom_0.serial_number,
                                atom_1_sn: atom_1.serial_number,
                                atom_0: i,
                                atom_1: j,
                                is_backbone: false,
                            });
                        }
                    }
                }
            }
        }

        self.common.build_adjacency_list();
        self.bonds_hydrogen = create_hydrogen_bonds_single_mol(
            &self.common.atoms,
            &self.common.atom_posits,
            &self.common.bonds,
        );
    }
}
