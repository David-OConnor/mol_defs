//! This defines a `MoleculeCommon` struct, which is shared by all molecule types. It includes
//! the most important features, like atoms, bonds, and metadata.
//!

use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::HashMap,
    f64::consts::PI,
    io,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use bio_files::BondType;
use dynamics::{find_planar_posit, find_tetra_posit_final, find_tetra_posits};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{Element, Element::Hydrogen};

// // Used by the mol editor, and alignment. Be careful with this!
// pub static NEXT_ATOM_SN: AtomicU32 = AtomicU32::new(0);
use crate::molecules::{Atom, Bond, MolIdent, build_adjacency_list};

#[derive(Clone, Copy, PartialEq)]
pub enum BondGeom {
    Linear,
    Planar,
    Tetrahedral,
}

/// Contains fields shared by all molecule types.
#[derive(Debug, Clone)]
pub struct MoleculeCommon {
    /// Note: This may have overlap with `MoleculeSmall::Ident` depending on
    /// how it's used.
    pub ident: String,
    /// A user settable name which defaults to None.
    pub name: Option<String>,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// A fast lookup for finding atoms, by index, covalently bonded to each atom.
    pub adjacency_list: Vec<Vec<usize>>,
    /// For repositioning atoms, e.g. from dynamics or absolute positioning.
    ///
    /// Absolute atom positions. For absolute conformation type[s], these positions are set and accessed directly, e.g., by MD
    /// simulations. We leave the molecule atom positions as ingested directly from data files. (e.g., relative positions).
    /// For rigid and semi-rigid conformations, these are derivative of the pose, in conjunction with
    /// the molecule atoms' (relative) positions.
    pub atom_posits: Vec<Vec3>,
    pub metadata: HashMap<String, String>,
    /// This is a bit different, as it's for our UI only. Doesn't fit with the others,
    /// but is safer and easier than trying to sync Vec indices.
    pub visible: bool,
    pub path: Option<PathBuf>,
    /// This is a cached derivative of `path`.
    pub filename: String,
    /// Inner valuje is the number of copies.
    pub selected_for_md: Option<usize>,
    pub entity_i_range: Option<(usize, usize)>,
    // todo: Consider if we should move this to MoleculeSmall etc.
    // todo: This index only/always applies to small molecules.
    /// We have instantiated multiple copies of a molecule for MD simulations.
    /// If Some, that is the index of the parent.
    pub copy_for_md: Option<usize>,
    /// A cache
    pub next_atom_sn: u32,
    /// The coordinates are a flat 2D depiction, e.g. PubChem's fallback for a compound with no
    /// 3D conformer. Set once on load from `posits_are_2d`, so the UI needn't reassess it; cleared
    /// by `make_3d`.
    pub is_2d: bool,
}

impl Default for MoleculeCommon {
    /// Only so we can set visible: true.
    fn default() -> Self {
        Self {
            ident: String::new(),
            name: None,
            metadata: HashMap::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            adjacency_list: Vec::new(),
            atom_posits: Vec::new(),
            visible: true,
            path: None,
            filename: String::new(),
            selected_for_md: None,
            entity_i_range: None,
            copy_for_md: None,
            next_atom_sn: 1,
            is_2d: false,
            // md_snapshot_range: None,
        }
    }
}

impl MoleculeCommon {
    /// If `bonds` is none, create it based on atom distances. Useful in the case of mmCIF files,
    /// which usually lack bond information.
    ///
    /// Hydrogens should have been added to the atom set, if required, prior to running this,
    /// so bonds are created.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let filename = match &path {
            Some(p) => p.file_stem().unwrap().to_string_lossy().to_string(),

            None => String::new(),
        };

        let mut result = Self {
            ident,
            metadata,
            atoms,
            bonds,
            atom_posits,
            path,
            filename,
            ..Self::default()
        };

        result.build_adjacency_list();
        result
    }

    pub fn get_atom(&self, i: usize) -> Option<&Atom> {
        if i < self.atoms.len() {
            Some(&self.atoms[i])
        } else {
            None
        }
    }

    pub fn get_atom_mut(&mut self, i: usize) -> Option<&mut Atom> {
        if i < self.atoms.len() {
            Some(&mut self.atoms[i])
        } else {
            None
        }
    }

    pub fn get_bond(&self, i: usize) -> Option<&Bond> {
        if i < self.bonds.len() {
            Some(&self.bonds[i])
        } else {
            None
        }
    }

    pub fn get_bond_mut(&mut self, i: usize) -> Option<&mut Bond> {
        if i < self.bonds.len() {
            Some(&mut self.bonds[i])
        } else {
            None
        }
    }

    pub fn update_path(&mut self, path: &Path) {
        self.path = Some(path.to_owned());
        self.filename = path.file_stem().unwrap().to_string_lossy().to_string();
    }

    /// Build a list of, for each atom, all atoms bonded to it.
    /// We use this as part of our flexible-bond conformation algorithm, and in setting up
    /// angles and dihedrals for molecular docking.
    ///
    /// Run this after populate hydrogens.
    pub fn build_adjacency_list(&mut self) {
        self.adjacency_list = build_adjacency_list(&self.bonds, self.atoms.len());
    }

    /// Largest bonded component within a selection, ranked by heavy atoms, then total atoms.
    /// Ties keep the first component in selection order. Returned indices retain that order.
    pub fn largest_connected_component(&self, indices: &[usize]) -> Vec<usize> {
        let mut remaining = vec![false; self.atoms.len()];
        for &i in indices {
            remaining[i] = true;
        }

        let mut largest = Vec::new();
        let mut best_score = (0, 0);
        for &start in indices {
            if !remaining[start] {
                continue;
            }

            remaining[start] = false;
            let mut stack = vec![start];
            let mut component = Vec::new();
            let mut heavy_count = 0;
            while let Some(i) = stack.pop() {
                component.push(i);
                heavy_count += usize::from(self.atoms[i].element != Hydrogen);

                for &neighbor in &self.adjacency_list[i] {
                    if remaining[neighbor] {
                        remaining[neighbor] = false;
                        stack.push(neighbor);
                    }
                }
            }

            let score = (heavy_count, component.len());
            if score > best_score {
                best_score = score;
                largest = component;
            }
        }

        let mut keep = vec![false; self.atoms.len()];
        for i in largest {
            keep[i] = true;
        }
        indices.iter().copied().filter(|&i| keep[i]).collect()
    }

    /// Group every atom into a bonded component: each returned `Vec` is one connected piece of
    /// this molecule's graph. A molecule ingested as a single unit usually yields one component,
    /// but salts, and selections carved out of a protein, may yield several.
    ///
    /// Ordered largest first, ranked by heavy atoms, then total atoms. Indices within a component
    /// are ascending.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut result = components_of(&self.adjacency_list);
        self.sort_components(&mut result);
        result
    }

    /// The bonded components this molecule would break into if the bonds at `bond_indices` were
    /// removed. Ordering matches [`Self::connected_components`], so the first entry is the piece
    /// to keep when splitting a molecule in place.
    ///
    /// Errors unless every bond given is a place the molecule comes apart. A ring is the usual
    /// cause: cutting one of its bonds leaves it joined the other way around, so splitting a ring
    /// takes two cuts.
    pub fn components_without_bonds(&self, bond_indices: &[usize]) -> io::Result<Vec<Vec<usize>>> {
        if bond_indices.is_empty() {
            return Err(invalid("No bonds were given to split at"));
        }

        let mut cut = vec![false; self.bonds.len()];
        for &i in bond_indices {
            if i >= self.bonds.len() {
                return Err(invalid(format!("Bond index {i} is out of range")));
            }
            cut[i] = true;
        }

        // Rebuild from the surviving bonds rather than editing `adjacency_list`: that list stores
        // only neighbor indices, so it can't tell which bond an entry came from if two bonds join
        // the same pair of atoms.
        let kept: Vec<Bond> = self
            .bonds
            .iter()
            .enumerate()
            .filter(|(i, _)| !cut[*i])
            .map(|(_, b)| b.clone())
            .collect();

        let mut result = components_of(&build_adjacency_list(&kept, self.atoms.len()));

        let mut comp_of = vec![0; self.atoms.len()];
        for (comp_i, component) in result.iter().enumerate() {
            for &i in component {
                comp_of[i] = comp_i;
            }
        }

        // Every bond given must be a place the molecule comes apart. Checking them one at a time,
        // rather than counting components, catches a mix of good and bad cuts, and names the one
        // at fault. (Counting wouldn't work anyway for a molecule already in several pieces.)
        for &i in bond_indices {
            let bond = &self.bonds[i];
            if comp_of[bond.atom_0] == comp_of[bond.atom_1] {
                return Err(invalid(format!(
                    "the bond between atoms {} and {} doesn't split the molecule: it's part of a \
                    ring, so cutting it leaves the ring joined the other way around. Select every \
                    bond that closes the ring, e.g. two of them",
                    bond.atom_0_sn, bond.atom_1_sn
                )));
            }
        }

        self.sort_components(&mut result);
        Ok(result)
    }

    /// Largest first, by heavy atoms, then total atoms. Ties keep their existing order.
    fn sort_components(&self, components: &mut [Vec<usize>]) {
        components.sort_by_key(|c| {
            let heavy = c
                .iter()
                .filter(|&&i| self.atoms[i].element != Hydrogen)
                .count();

            (Reverse(heavy), Reverse(c.len()))
        });
    }

    /// A standalone copy of the atoms at `atom_indices`, and of the bonds with both ends among
    /// them. Serial numbers are re-assigned from 1, and each atom keeps its current (moved)
    /// position, as both its internal position and in `atom_posits`.
    ///
    /// E.g. for pulling a fragment out of a molecule as its own molecule. Pass a component from
    /// [`Self::components_without_bonds`] to keep that fragment whole.
    pub fn subset(&self, atom_indices: &[usize]) -> io::Result<Self> {
        if atom_indices.is_empty() {
            return Err(invalid("No atoms were given to copy"));
        }

        // Old index -> (new index, new SN). Used to rebuild bonds.
        let mut map = HashMap::with_capacity(atom_indices.len());
        let mut atoms = Vec::with_capacity(atom_indices.len());

        for (i_new, &i_old) in atom_indices.iter().enumerate() {
            let atom = self
                .atoms
                .get(i_old)
                .ok_or_else(|| invalid(format!("Atom index {i_old} is out of range")))?;

            let sn = i_new as u32 + 1;
            if map.insert(i_old, (i_new, sn)).is_some() {
                return Err(invalid(format!("Atom index {i_old} was given twice")));
            }

            atoms.push(Atom {
                serial_number: sn,
                // The caller is pulling this fragment out where it sits; the parent's internal
                // positions may be a different (e.g. pre-move) frame.
                posit: self.atom_posits[i_old],
                residue: None,
                chain: None,
                ..atom.clone()
            });
        }

        let mut bonds = Vec::new();
        for bond in &self.bonds {
            let (Some(&(atom_0, atom_0_sn)), Some(&(atom_1, atom_1_sn))) =
                (map.get(&bond.atom_0), map.get(&bond.atom_1))
            else {
                continue;
            };

            bonds.push(Bond {
                bond_type: bond.bond_type,
                atom_0_sn,
                atom_1_sn,
                atom_0,
                atom_1,
                is_backbone: false,
            });
        }

        Ok(Self::new(
            self.ident.clone(),
            atoms,
            bonds,
            self.metadata.clone(),
            None,
        ))
    }

    /// Remove a set of atoms, and every bond to them, in one pass. Equivalent to repeated
    /// [`Self::remove_atom`] calls, but re-indexes once instead of once per atom.
    pub fn remove_atoms(&mut self, indices: &[usize]) {
        let mut remove = vec![false; self.atoms.len()];
        for &i in indices {
            if i >= self.atoms.len() {
                eprintln!("Error removing atoms: Index {i} out of range");
                return;
            }
            remove[i] = true;
        }

        // Where each atom ends up once the removals are applied.
        let mut i_new = Vec::with_capacity(self.atoms.len());
        let mut kept = 0;
        for i in 0..self.atoms.len() {
            i_new.push(kept);
            if !remove[i] {
                kept += 1;
            }
        }

        let mut i = 0;
        self.atoms.retain(|_| {
            let keep = !remove[i];
            i += 1;
            keep
        });

        let mut i = 0;
        self.atom_posits.retain(|_| {
            let keep = !remove[i];
            i += 1;
            keep
        });

        self.bonds.retain_mut(|bond| {
            if remove[bond.atom_0] || remove[bond.atom_1] {
                return false;
            }

            bond.atom_0 = i_new[bond.atom_0];
            bond.atom_1 = i_new[bond.atom_1];
            true
        });

        self.build_adjacency_list();
    }

    /// If every atom's internal position has Z = 0, as in a 2D depiction. SDF and Mol2 files
    /// write 4 decimal places, so a real Z offset of any size clears the threshold.
    ///
    /// Note that a planar 3D structure, e.g. CO2, can pass this if it lies in the XY plane.
    /// Run this before adding hydrogens: they're placed with 3D geometry, even around flat atoms.
    pub fn posits_are_2d(&self) -> bool {
        const Z_MAX: f64 = 1e-5;

        // One or two atoms have no 3D geometry to build.
        self.atoms.len() >= 3 && self.atoms.iter().all(|a| a.posit.z.abs() < Z_MAX)
    }

    /// Reset atom positions to be at their internal values, e.g. as present in the Mol2 or SDF files.
    pub fn reset_posits(&mut self) {
        self.atom_posits = self.atoms.iter().map(|a| a.posit).collect();
    }

    /// Update local positions so they're centered around the origin. Useful for molecule creation
    /// workflows.
    pub fn center_local_posits_around_origin(&mut self) {
        // Same logic as `centroid`, but for local positions.
        let center = {
            let mut c = Vec3::new_zero();
            for atom in &self.atoms {
                c += atom.posit;
            }
            c / self.atoms.len() as f64
        };

        for atom in &mut self.atoms {
            atom.posit -= center;
        }
    }

    /// Used for rotation and motion; the rough center of the molecule.
    pub fn centroid(&self) -> Vec3 {
        let n = self.atom_posits.len() as f64;
        let sum = self
            .atom_posits
            .iter()
            .fold(Vec3::new_zero(), |a, b| a + *b);
        sum / n
    }

    /// Uses atom internal positions.
    pub fn centroid_local(&self) -> Vec3 {
        let n = self.atoms.len() as f64;
        let mut sum = Vec3::new_zero();

        for a in &self.atoms {
            sum += a.posit;
        }

        sum / n
    }

    pub fn move_to(&mut self, pos: Vec3) {
        let delta = pos - self.centroid();
        for posit in &mut self.atom_posits {
            *posit += delta;
        }
    }

    pub fn shift(&mut self, delta: Vec3) {
        for posit in &mut self.atom_posits {
            *posit += delta;
        }
    }

    pub fn rotate(&mut self, rot: Quaternion, pivot_: Option<usize>) {
        let pivot = match pivot_ {
            Some(i) => self.atom_posits[i],
            None => self.centroid(),
        };

        for posit in &mut self.atom_posits {
            let local = *posit - pivot;
            let rotated = rot.rotate_vec(local);
            let out = rotated + pivot;

            *posit = out;
        }
    }

    /// Removes an atom, and any bond to it. Re-index bonds due to this
    /// removal from likely the interior of the molecule's seq.
    pub fn remove_atom(&mut self, i: usize) {
        if i >= self.atoms.len() {
            eprintln!("Error removing atom: Out of range");
            return;
        }

        self.atoms.remove(i);
        self.atom_posits.remove(i);

        self.bonds.retain_mut(|bond| {
            if bond.atom_0 == i || bond.atom_1 == i {
                return false;
            }

            if bond.atom_0 > i {
                bond.atom_0 -= 1;
            }
            if bond.atom_1 > i {
                bond.atom_1 -= 1;
            }
            true
        });

        self.adjacency_list.remove(i);

        for adj in &mut self.adjacency_list {
            adj.retain(|&j| j != i);

            for j in adj.iter_mut() {
                if *j > i {
                    *j -= 1;
                }
            }
        }
    }

    /// Re-assign atom serial numbers as 1-ripple. Useful after or during editing, especially
    /// prior to saving in SDF format, which doesn't explicitly list SNs with the atom.
    /// We also  use it when assembling nucleic acids and other molecule generation.
    pub fn reassign_sns(&mut self) {
        // todo: Be more clever about this.
        let mut updated_sns = Vec::with_capacity(self.atoms.len());

        for (i, atom) in self.atoms.iter_mut().enumerate() {
            let sn_new = i as u32 + 1;
            atom.serial_number = sn_new;
            updated_sns.push(sn_new);
        }

        for bond in &mut self.bonds {
            bond.atom_0_sn = updated_sns[bond.atom_0];
            bond.atom_1_sn = updated_sns[bond.atom_1];
        }

        self.next_atom_sn = match updated_sns.last() {
            Some(l) => *l + 1,
            None => 1,
        };
    }

    /// The sum of each atom's elemental atomic weight, in Daltons (amu).
    pub fn atomic_weight(&self) -> f32 {
        let result: f64 = self
            .atoms
            .iter()
            .map(|a| a.element.atomic_weight() as f64)
            .sum();

        result as f32
    }

    /// Unweighted chemistry adjacency matrix: A N×N matrix with 1 where a bond exists
    /// (0 otherwise). N is the atom count.
    pub fn adjacency_matrix(&self) -> Vec<Vec<u8>> {
        let n = self.adjacency_list.len();
        let mut result = vec![vec![0; n]; n];

        for (i, neighs) in self.adjacency_list.iter().enumerate() {
            for &j in neighs {
                if j < n && i != j {
                    result[i][j] = 1;
                    result[j][i] = 1;
                }
            }
        }

        result
    }

    /// Filename prefixes used when caching molecules downloaded from an online source. See
    /// `name()`.
    const MANAGED_FILENAME_PREFIXES: [&str; 7] = [
        "pubchem-",
        "chebi-",
        "drugbank-",
        "rcsb-",
        "geostd-",
        "smiles-",
        "built-in-",
    ];

    /// Uses the custom name when set. Otherwise combines `ident` with the PubChem title,
    /// when available, and a distinct filename. The title or filename can provide a more
    /// useful description than a CID alone.
    pub fn name(&self, idents: Option<&Vec<MolIdent>>) -> Cow<'_, str> {
        if let Some(name) = self.name.as_deref() {
            return Cow::Borrowed(name);
        }

        let mut result = self.ident.to_string();

        if let Some(idents_) = idents {
            // Alternatively, `MolIdent::IupacName` ident will also work.
            for ident in idents_ {
                if let MolIdent::PubchemTitle(t) = ident {
                    result.push_str(&format!(" | {t}"));
                    break;
                }
            }
        }

        let filename = self.filename.to_lowercase();
        let filename = filename.trim();

        // Molecules downloaded from an API, and not explicitly saved by the user, are cached under
        // a "provider-key" filename, e.g. "pubchem-3672". The key is generally the identifier we
        // already display, so compare against the key alone to avoid repeating it.
        let filename_key = Self::MANAGED_FILENAME_PREFIXES
            .iter()
            .find_map(|prefix| filename.strip_prefix(prefix))
            .unwrap_or(filename);

        // These checks prevent a duplicate if the filename is effectively the identifier, or the
        // PubChem title.
        if !filename_key.is_empty() && !result.to_lowercase().contains(filename_key) {
            // Don't show the full filename if it's long.
            let truncated = if self.filename.chars().count() > 12 {
                let mut s: String = self.filename.chars().take(12).collect();
                s.push_str("...");
                s
            } else {
                self.filename.clone()
            };

            result.push_str(&format!(" | {truncated}"));
        }

        Cow::Owned(result)
    }

    /// A helper used to ensure that there is a valid atom for each bond. (Checks both SN and index),
    /// and that checks if the adjacency list is up to date. This is used for debugging only.
    #[allow(unused)]
    pub fn validate_bonds(&self) {
        println!("\nValidating bonds... (This should not be in permanent code)\n");
        for bond in &self.bonds {
            assert!(bond.atom_0 < self.atoms.len());
            assert!(bond.atom_1 < self.atoms.len());
            assert_ne!(bond.atom_0, bond.atom_1);

            assert!(self.adjacency_list[bond.atom_0].contains(&bond.atom_1));
            assert!(self.adjacency_list[bond.atom_1].contains(&bond.atom_0));

            assert_eq!(self.adjacency_list.len(), self.atoms.len());

            assert_eq!(bond.atom_0_sn, self.atoms[bond.atom_0].serial_number);
            assert_eq!(bond.atom_1_sn, self.atoms[bond.atom_1].serial_number);
        }
    }

    /// Adds an atom, and a bond between it and an existing one [parent]. Also adds Hydrogens on this atom.
    /// Returns (atom's new index, bond's new index).
    pub fn add_atom(
        &mut self,
        i_par: usize,
        element: Element,
        bond_type: BondType,
        ff_type: Option<String>,
        bond_len: Option<f64>,
        q: Option<f32>,
    ) -> Option<(usize, usize)> {
        let el_parent = self.atoms[i_par].element;

        if el_parent == Hydrogen {
            return None;
        }

        // Delete hydrogens; we'll add back if required.
        let par_sn = self.atoms[i_par].serial_number;
        if element != Hydrogen {
            remove_hydrogens(self, i_par);
        }

        // Removing hydrogens shifts the index of every atom that followed them, so the caller's
        // `i_par` is only valid if all of this atom's H happen to come after it in the list.
        // Serial numbers are stable across removals; re-derive the index from ours.
        let i_par = self.atoms.iter().position(|a| a.serial_number == par_sn)?;
        let posit_parent = self.atom_posits[i_par];

        // Geometry comes from the bond orders this atom carries, not from its free valence: lone
        // pairs occupy coordination sites too, so an ether O (two single bonds) is bent like an
        // sp3 carbon rather than linear, and an amine N is pyramidal rather than trigonal planar.
        let geom = geom_for_atom(i_par, &self.bonds);

        // todo: Can't use `common` below here due to the delete_atom code and ownership.
        let posit = find_appended_posit(
            posit_parent,
            &self.atoms,
            &self.adjacency_list[i_par],
            bond_len,
            element,
            geom,
        )?;

        let new_sn = self.next_atom_sn;
        self.next_atom_sn += 1;

        let i_new_atom = self.atoms.len();
        let i_new_bond = self.bonds.len();

        if i_par >= self.atoms.len() {
            eprintln!("Index out of range when adding atoms: {i_par}");
            return None;
            // todo: This return and print are a workaround; find the root cause.
        }

        let atom_new = Atom {
            serial_number: new_sn,
            posit,
            element: element.clone(),
            type_in_res: None,
            force_field_type: ff_type.clone(),
            partial_charge: q,
            ..Default::default()
        };

        self.atoms.push(atom_new);

        self.atom_posits.push(posit);
        self.adjacency_list[i_par].push(i_new_atom);
        self.adjacency_list.push(vec![i_par]);

        self.bonds.push(Bond {
            bond_type,
            atom_0_sn: self.atoms[i_par].serial_number,
            atom_1_sn: new_sn,
            atom_0: i_par,
            atom_1: i_new_atom,
            is_backbone: false,
        });

        Some((i_new_atom, i_new_bond))
    }

    /// Populate  hydrogens like the standalone editor fn, but only update the mol; no drawing/state
    /// updates  etc. We use this, for example, when loading molecules that don't hav eH.
    pub fn populate_hydrogens_on_atom(&mut self, i: usize) {
        // todo: Dry with the other fn.
        if i >= self.atoms.len() {
            eprintln!("Error: Invalid atom index when populating Hydrogens.");
            return;
        }

        let el = self.atoms[i].element;
        if el == Hydrogen {
            return;
        }

        let h_to_add = bonds_avail(i, self, el);

        // let bonds_remaining = bonds_avail.saturating_sub(adj.len());

        for _ in 0..h_to_add {
            let atom = &self.atoms[i];
            let (ff_type, bond_len) = {
                let mut v = (None, 1.1);

                // Grabbing the first, arbitrarily.
                for (ff, bl) in hydrogens_avail(&atom.force_field_type) {
                    v.0 = Some(ff);
                    v.1 = bl;
                    break;
                }

                v
            };

            // I believe we populate partial charge after  and ff  type  after?
            self.add_atom(i, Hydrogen, BondType::Single, ff_type, Some(bond_len), None);
        }
    }

    pub fn populate_hydrogens(&mut self) {
        self.update_next_sn();
        for i in 0..self.atoms.len() {
            self.populate_hydrogens_on_atom(i);
        }
    }

    pub fn update_next_sn(&mut self) {
        let mut highest_sn = 0;
        for atom in &self.atoms {
            if atom.serial_number > highest_sn {
                highest_sn = atom.serial_number;
            }
        }

        self.next_atom_sn = highest_sn + 1;
    }
}

/// Given stable atom serial numbers, reassign bond indices to match. Useful, for example, after
/// filtering a set of atoms and  bonds.
pub fn reassign_bond_indices(bonds: &mut [Bond], atoms: &[Atom]) {
    let sn_to_new_i: HashMap<_, _> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i)) // use usize if your bond fields are usize
        .collect();

    for b in bonds {
        b.atom_0 = sn_to_new_i[&b.atom_0_sn];
        b.atom_1 = sn_to_new_i[&b.atom_1_sn];
    }
}

pub fn find_appended_posit(
    posit_parent: Vec3,
    atoms: &[Atom],
    adj_to_par: &[usize],
    bond_len: Option<f64>,
    element: Element,
    geom: BondGeom,
) -> Option<Vec3> {
    let neighbor_count = adj_to_par.len();

    // Note on these computations: The parent atom is the "hub" of a tetrahedral or planar
    // hub-and-spoke config. Other spokes are existing atoms bound to this parent, and the atom
    // we're computing the position here to add.
    let result = match neighbor_count {
        // This 0 branch should only be called for disconnected parents.
        0 => Some(posit_parent + Vec3::new(1.3, 0., 0.)),
        1 => {
            // The single placed neighbor is the *grandparent* direction from the parent.
            // Use the geometry-appropriate bond angle so that sp (linear), sp2 (planar)
            // and sp3 (tetrahedral) atoms all get the correct valence angle.
            let grandparent = atoms[adj_to_par[0]].posit;

            const TETRA_ANGLE: f64 = 1.91063; // 109.47°
            const PLANAR_ANGLE: f64 = 2.0 * PI / 3.0; // 120.00°

            let bond_par_gp = (grandparent - posit_parent).to_normalized();

            match geom {
                BondGeom::Linear => {
                    // Place directly opposite the grandparent (180°).
                    Some(posit_parent + (-bond_par_gp))
                }
                BondGeom::Planar => {
                    let ax_rot = bond_par_gp.any_perpendicular();
                    let rotator = Quaternion::from_axis_angle(ax_rot, PLANAR_ANGLE);
                    let relative_dir = rotator.rotate_vec(bond_par_gp);
                    Some(posit_parent + relative_dir)
                }
                BondGeom::Tetrahedral => {
                    let ax_rot = bond_par_gp.any_perpendicular();
                    let rotator = Quaternion::from_axis_angle(ax_rot, TETRA_ANGLE);
                    // If H, shorten the bond (only matters when bond_len is None).
                    let mut relative_dir = rotator.rotate_vec(bond_par_gp);
                    if element == Hydrogen {
                        relative_dir = (relative_dir.to_normalized()) * 1.1;
                    }
                    Some(posit_parent + relative_dir)
                }
            }
        }
        2 => {
            let neighbor_0 = atoms[adj_to_par[0]].posit;
            let neighbor_1 = atoms[adj_to_par[1]].posit;

            match geom {
                BondGeom::Tetrahedral => {
                    // This function uses the distance between the first two params, so it's likely
                    // in the case of adding H, this is what we want. (?)
                    let (p0, p1) = find_tetra_posits(posit_parent, neighbor_1, neighbor_0);

                    // Score a candidate by its minimum distance to any existing neighbor; pick the larger score.
                    let neighbors: &[usize] = &adj_to_par;
                    let score = |p: Vec3| {
                        let mut best = f64::INFINITY;
                        for &ni in neighbors {
                            let q = atoms[ni].posit;
                            let d2 = (p - q).magnitude_squared();
                            if d2 < best {
                                best = d2;
                            }
                        }
                        best
                    };

                    Some(if score(p0) >= score(p1) { p0 } else { p1 })
                }
                BondGeom::Planar => Some(find_planar_posit(posit_parent, neighbor_0, neighbor_1)),
                BondGeom::Linear => {
                    return None;
                }
            }
        }
        3 => {
            if geom != BondGeom::Tetrahedral {
                return None;
            }

            // None
            let adj_0 = adj_to_par[0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_to_par[1];
            let neighbor_1 = atoms[adj_1].posit;
            let adj_2 = adj_to_par[2];
            let neighbor_2 = atoms[adj_2].posit;

            // todo. Check both angles?
            // If the incoming angles are ~τ/3, add in a planar config.
            // let bond_0 = neighbor_0 - posit_parent;
            // let bond_1 = neighbor_1 - posit_parent;
            // let angle = bond_1.to_normalized().dot(bond_0.to_normalized()).acos();

            // Planar; full.
            // if angle > 1.95 {
            // todo: Experiment. You may wish to use the character of neighboring bond count
            // todo and type instead of this angle.
            // if angle > 2.11 {
            //     println!("Planar abort!: {angle}"); // todo temp!!
            //     return None;
            // } else {
            Some(find_tetra_posit_final(
                posit_parent,
                neighbor_0,
                neighbor_1,
                neighbor_2,
            ))
            // }
        }
        _ => None,
    };

    // Set len, if applicable.
    // todo: Could be slightly more efficient to bake this length correction into the find_tetra
    // todo etc fns.
    match result {
        Some(p) => match bond_len {
            Some(l) => {
                let rel_pos = (p - posit_parent).to_normalized() * l;
                Some(posit_parent + rel_pos)
            }
            None => Some(p),
        },
        None => None,
    }
}

/// The coordination geometry around an atom, from the bond orders it carries. Note that this
/// counts lone pairs implicitly: an atom with only single bonds is tetrahedral whatever its
/// element, so an ether O comes out bent (~109°) rather than linear.
pub fn geom_for_atom(i: usize, bonds: &[Bond]) -> BondGeom {
    let atom_bonds: Vec<&Bond> = bonds
        .iter()
        .filter(|b| b.atom_0 == i || b.atom_1 == i)
        .collect();

    if atom_bonds.iter().any(|b| b.bond_type == BondType::Triple) {
        return BondGeom::Linear;
    }

    // Cumulated diene (allene-type, e.g. C=C=C): the central atom carries two
    // double bonds and is sp-hybridised (linear), not sp2.
    let double_count = atom_bonds
        .iter()
        .filter(|b| b.bond_type == BondType::Double)
        .count();
    if double_count >= 2 {
        return BondGeom::Linear;
    }

    if atom_bonds
        .iter()
        .any(|b| matches!(b.bond_type, BondType::Double | BondType::Aromatic))
    {
        BondGeom::Planar
    } else {
        BondGeom::Tetrahedral
    }
}

/// The bond order an atom already carries, doubled so aromatic bonds (order 1.5) stay in integer
/// arithmetic. A benzene carbon's two aromatic bonds come to 6 here, i.e. an order of 3.
fn bond_order_x2(i_atom: usize, bonds: &[Bond]) -> isize {
    let mut result = 0;

    for bond in bonds {
        if bond.atom_0 != i_atom && bond.atom_1 != i_atom {
            continue;
        }

        result += match bond.bond_type {
            BondType::Single => 2,
            BondType::Double => 4,
            BondType::Triple => 6,
            BondType::Aromatic => 3,
            _ => 2,
        };
    }

    result
}

/// How many more single bonds (in practice, Hydrogens) an atom can accept: its typical valence,
/// less the bond order it already carries.
///
/// This assumes a neutral atom in its usual valence state. Charged centres -- an ammonium N, the
/// terminal N of an azide, a carboxylate O -- don't follow it, so the molecule editor lets the
/// user override the Hydrogen count directly rather than trying to infer a formal charge here.
pub fn bonds_avail(i_atom: usize, mol: &MoleculeCommon, el: Element) -> usize {
    use Element::*;

    let order_x2 = bond_order_x2(i_atom, &mol.bonds);
    // Round up: a fused-ring carbon carrying three aromatic bonds (order 4.5) is full, not
    // short half a bond.
    let order = (order_x2 + 1) / 2;

    let valence: isize = match el {
        Hydrogen => 1,
        Carbon | Silicon => 4,
        Nitrogen | Boron => 3,
        Oxygen => 2,
        Fluorine | Chlorine | Bromine | Iodine => 1,
        // S and P routinely exceed their base valence (sulfoxides, sulfones, phosphates), so step
        // up to the next state that accommodates the bonds already present.
        Sulfur | Selenium | Tellurium => *[2, 4, 6].iter().find(|v| **v >= order).unwrap_or(&6),
        Phosphorus => *[3, 5].iter().find(|v| **v >= order).unwrap_or(&5),
        // Metals and anything else we don't model: leave alone rather than guess at Hydrogens.
        _ => return 0,
    };

    (valence - order).max(0) as usize
}

/// Remove all hydrogens bonded to an atom.
pub fn remove_hydrogens(mol: &mut MoleculeCommon, i: usize) {
    let mut h_to_del = Vec::new();

    // Remove Hydrogens; we'll add any back as applicable.
    for j in &mol.adjacency_list[i] {
        if mol.atoms[*j].element == Hydrogen {
            h_to_del.push(*j);
        }
    }

    h_to_del.sort_unstable_by(|a, b| b.cmp(a));
    for j in h_to_del {
        mol.remove_atom(j);
    }
}

// todo: I think this approach is wrong. You can add multiple of the same one...
/// This is built from Amber's gaff2.dat. Returns each H FF type that can be bound to a given atom
/// (by force field type), and the bond distance in Å.
pub fn hydrogens_avail(ff_type: &Option<String>) -> Vec<(String, f64)> {
    let Some(f) = ff_type else { return Vec::new() };
    match f.as_ref() {
        // Water
        "ow" => vec![("hw".to_owned(), 0.9572)],
        "hw" => vec![("hw".to_owned(), 1.5136)],

        // Generic sp carbon (c )
        "c" => vec![
            ("h4".to_owned(), 1.1123),
            ("h5".to_owned(), 1.1053),
            ("ha".to_owned(), 1.1010),
        ],

        // sp2 carbon families
        "c1" => vec![("ha".to_owned(), 1.0666), ("hc".to_owned(), 1.0600)],
        "c2" => vec![
            ("h4".to_owned(), 1.0865),
            ("h5".to_owned(), 1.0908),
            ("ha".to_owned(), 1.0882),
            ("hc".to_owned(), 1.0870),
            ("hx".to_owned(), 1.0836),
        ],
        "c3" => vec![
            ("h1".to_owned(), 1.0969),
            ("h2".to_owned(), 1.0950),
            ("h3".to_owned(), 1.0938),
            ("hc".to_owned(), 1.0962),
            ("hx".to_owned(), 1.0911),
        ],
        "c5" => vec![
            ("h1".to_owned(), 1.0972),
            ("h2".to_owned(), 1.0955),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0954),
            ("hx".to_owned(), 1.0917),
        ],
        "c6" => vec![
            ("h1".to_owned(), 1.0984),
            ("h2".to_owned(), 1.0985),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0979),
            ("hx".to_owned(), 1.0931),
        ],

        // Aromatic/condensed ring carbons
        "ca" => vec![
            ("ha".to_owned(), 1.0860),
            ("h4".to_owned(), 1.0885),
            ("h5".to_owned(), 1.0880),
        ],
        "cc" => vec![
            ("h4".to_owned(), 1.0809),
            ("h5".to_owned(), 1.0820),
            ("ha".to_owned(), 1.0838),
            ("hx".to_owned(), 1.0827),
        ],
        "cd" => vec![
            ("h4".to_owned(), 1.0818),
            ("h5".to_owned(), 1.0821),
            ("ha".to_owned(), 1.0835),
            ("hx".to_owned(), 1.0801),
        ],
        "ce" => vec![
            ("h4".to_owned(), 1.0914),
            ("h5".to_owned(), 1.0895),
            ("ha".to_owned(), 1.0880),
        ],
        "cf" => vec![
            ("h4".to_owned(), 1.0942),
            ("ha".to_owned(), 1.0885),
            // table also lists h5-cf (reverse order) at 1.0890
            ("h5".to_owned(), 1.0890),
        ],
        "cg" => Vec::new(), // no H entries shown for cg in the provided snippet

        // Other carbon families frequently seen
        "cu" => vec![("ha".to_owned(), 1.0786)],
        "cv" => vec![("ha".to_owned(), 1.0878)],
        "cx" => vec![
            ("h1".to_owned(), 1.0888),
            ("h2".to_owned(), 1.0869),
            ("hc".to_owned(), 1.0865),
            ("hx".to_owned(), 1.0849),
        ],
        "cy" => vec![
            ("h1".to_owned(), 1.0946),
            ("h2".to_owned(), 1.0930),
            ("hc".to_owned(), 1.0947),
            ("hx".to_owned(), 1.0913),
        ],

        // Nitrogen families: protonated H type is "hn"
        "n1" => vec![("hn".to_owned(), 0.9860)],
        "n2" => vec![("hn".to_owned(), 1.0221)],
        "n3" => vec![("hn".to_owned(), 1.0190)],
        "n4" => vec![("hn".to_owned(), 1.0300)],
        "n" => vec![("hn".to_owned(), 1.0130)],
        "n5" => vec![("hn".to_owned(), 1.0211)],
        "n6" => vec![("hn".to_owned(), 1.0183)],
        "n7" => vec![("hn".to_owned(), 1.0195)],
        "n8" => vec![("hn".to_owned(), 1.0192)],
        "n9" => vec![("hn".to_owned(), 1.0192)],
        "na" => vec![("hn".to_owned(), 1.0095)],
        "nh" => vec![("hn".to_owned(), 1.0120)],
        "nj" => vec![("hn".to_owned(), 1.0130)],
        "nl" => vec![("hn".to_owned(), 1.0476)],
        "no" => vec![("hn".to_owned(), 1.0440)],
        "np" => vec![("hn".to_owned(), 1.0210)],
        "nq" => vec![("hn".to_owned(), 1.0180)],
        "ns" => vec![("hn".to_owned(), 1.0132)],
        "nt" => vec![("hn".to_owned(), 1.0105)],
        "nu" => vec![("hn".to_owned(), 1.0137)],
        "nv" => vec![("hn".to_owned(), 1.0114)],
        "nx" => vec![("hn".to_owned(), 1.0338)],
        "ny" => vec![("hn".to_owned(), 1.0339)],
        "nz" => vec![("hn".to_owned(), 1.0271)],

        // Oxygen families: hydroxyl H type is "ho"
        "o" => vec![("ho".to_owned(), 0.9810)],
        "oh" => vec![("ho".to_owned(), 0.9725)],

        // Sulfur families: thiol H type is "hs"
        "s" => vec![("hs".to_owned(), 1.3530)],
        "s4" => vec![("hs".to_owned(), 1.3928)],
        "s6" => vec![("hs".to_owned(), 1.3709)],
        "sh" => vec![("hs".to_owned(), 1.3503)],
        "sy" => vec![("hs".to_owned(), 1.3716)],

        // Phosphorus families: acidic phosphate H type is "hp"
        "p2" => vec![("hp".to_owned(), 1.4272)],
        "p3" => vec![("hp".to_owned(), 1.4256)],
        "p4" => vec![("hp".to_owned(), 1.4271)],
        "p5" => vec![("hp".to_owned(), 1.4205)],
        "py" => vec![("hp".to_owned(), 1.4150)],

        _ => Vec::new(),
    }
}

/// Flood-fill an adjacency list into its connected components; ascending index order within each.
/// See [`MoleculeCommon::connected_components`].
fn components_of(adj_list: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; adj_list.len()];
    let mut result = Vec::new();

    for start in 0..adj_list.len() {
        if visited[start] {
            continue;
        }

        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();

        while let Some(i) = stack.pop() {
            component.push(i);

            for &neighbor in &adj_list[i] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        component.sort_unstable();
        result.push(component);
    }

    result
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use bio_files::BondType;
    use na_seq::Element;

    use super::*;

    /// A chain of carbons, bonded `atoms[i]`—`atoms[i + 1]`, plus any extra bonds given as
    /// (atom, atom) pairs; e.g. `(0, 4)` closes the first five into a ring.
    fn chain(len: usize, extra: &[(usize, usize)]) -> MoleculeCommon {
        let atoms: Vec<_> = (0..len)
            .map(|i| Atom {
                serial_number: i as u32 + 1,
                posit: Vec3::new(i as f64 * 1.5, 0., 0.),
                element: Element::Carbon,
                ..Default::default()
            })
            .collect();

        let bonds = (0..len - 1)
            .map(|i| (i, i + 1))
            .chain(extra.iter().copied())
            .map(|(atom_0, atom_1)| Bond {
                bond_type: BondType::Single,
                atom_0_sn: atom_0 as u32 + 1,
                atom_1_sn: atom_1 as u32 + 1,
                atom_0,
                atom_1,
                is_backbone: false,
            })
            .collect();

        MoleculeCommon::new(String::from("TEST"), atoms, bonds, HashMap::new(), None)
    }

    #[test]
    fn splits_a_chain_at_one_bond() {
        // Cutting 2—3 of a 6-chain leaves 3 atoms on each side; an even split keeps them in
        // their original order.
        let components = chain(6, &[]).components_without_bonds(&[2]).unwrap();

        assert_eq!(components, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn rejects_a_single_ring_bond() {
        // A ring of 5, with a 2-atom tail. Cutting one of the ring's bonds leaves it joined the
        // other way around, so nothing comes apart.
        assert!(chain(7, &[(0, 4)]).components_without_bonds(&[0]).is_err());

        // Both of the ring's cuts: the ring opens, and the tail goes with one of the halves.
        let components = chain(7, &[(0, 4)])
            .components_without_bonds(&[0, 3])
            .unwrap();
        assert_eq!(components, vec![vec![0, 4, 5, 6], vec![1, 2, 3]]);
    }

    #[test]
    fn rejects_a_ring_bond_mixed_with_a_good_cut() {
        // Bond 5 (the tail) does split it; bond 0 (in the ring) doesn't. Counting components
        // would miss this, since the count still goes up.
        assert!(
            chain(7, &[(0, 4)])
                .components_without_bonds(&[0, 5])
                .is_err()
        );
    }

    #[test]
    fn keeps_existing_fragments_separate() {
        // Two chains of 3 with no bond between them, so the molecule starts out in two pieces.
        let mut mol = chain(6, &[]);
        mol.bonds.remove(2);
        mol.build_adjacency_list();

        assert_eq!(mol.connected_components().len(), 2);

        // Cutting 0—1 leaves three pieces, not two: the piece that was already separate stays so.
        let components = mol.components_without_bonds(&[0]).unwrap();
        assert_eq!(components, vec![vec![3, 4, 5], vec![1, 2], vec![0]]);
    }

    #[test]
    fn subset_carries_bonds_and_moved_positions() {
        let mut mol = chain(6, &[]);
        mol.shift(Vec3::new(0., 10., 0.));

        let sub = mol.subset(&[3, 4, 5]).unwrap();

        assert_eq!(sub.atoms.len(), 3);
        // The two bonds within the subset come along; 2—3, which leaves it, does not.
        assert_eq!(sub.bonds.len(), 2);
        assert_eq!(sub.atoms[0].serial_number, 1);
        assert_eq!(sub.bonds[0].atom_0_sn, 1);
        assert_eq!(sub.atoms[0].posit, mol.atom_posits[3]);
        assert_eq!(sub.atom_posits[0], mol.atom_posits[3]);
    }

    #[test]
    fn remove_atoms_reindexes_bonds() {
        let mut mol = chain(6, &[]);
        mol.remove_atoms(&[0, 1, 2]);

        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.atom_posits.len(), 3);
        assert_eq!(mol.bonds.len(), 2);
        assert_eq!((mol.bonds[0].atom_0, mol.bonds[0].atom_1), (0, 1));
        assert_eq!(mol.adjacency_list, vec![vec![1], vec![0, 2], vec![1]]);
    }
}
