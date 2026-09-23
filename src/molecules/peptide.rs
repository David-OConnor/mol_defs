/// Proteins / polypeptides
use std::collections::{HashMap, HashSet};
use std::{
    io,
    io::ErrorKind,
    path::PathBuf,
    sync::{mpsc, mpsc::Receiver},
    thread,
    time::Instant,
};

use bio_apis::{
    ReqError,
    pdbe::SiftsUniprotMapping,
    rcsb,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::{BackboneSS, DensityMap, ExperimentalMethod, MmCif, ResidueType, create_bonds};
use dynamics::{
    params::{ProtFfChargeMapSet, prepare_peptide_mmcif},
    populate_hydrogens_dihedrals,
};
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, Element};

use crate::{
    bond_inference::create_hydrogen_bonds_single_mol,
    molecules,
    molecules::{
        Atom, AtomRole, Bond, Chain, HydrogenBond, PeptideIdent, Residue, common::MoleculeCommon,
    },
    reflection::{DensityPt, DensityRect, ReflectionsData},
    util::mol_center_size,
};

/// A polypeptide molecule, e.g. a protein.
#[derive(Debug, Default, Clone)]
pub struct MoleculePeptide {
    pub common: MoleculeCommon,
    pub idents: Vec<PeptideIdent>,
    pub bonds_hydrogen: Vec<HydrogenBond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
    /// We currently use this for aligning ligands to CIF etc data, where they may already be included
    /// in a protein/ligand complex as hetero atoms.
    pub het_residues: Vec<Residue>,
    // /// Solvent-accessible surface. Used as one of our visualization methods.
    // /// Current structure is a Vec of rings.
    // /// Initializes to empty; updated A/R when the appropriate view is selected.
    // pub sa_surface_pts: Option<Vec<Vec<Vec3F32>>>,
    pub secondary_structure: Vec<BackboneSS>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    /// The full (Or partial while WIP) results from the RCSB data api.
    pub rcsb_data: Option<PdbDataResults>,
    pub rcsb_files_avail: Option<FilesAvailable>,
    pub reflections_data: Option<ReflectionsData>,
    /// This is the processed collection of electron density points, ready to be mapped
    /// to entities, with some amplitude processing. It not not explicitly grid or unit-cell based,
    /// although it was likely created from unit cell data.
    /// E.g. from a MAP or MTX file directly, or processed from raw reflections data
    /// in a 2fo-fc file.
    pub elec_density: Option<Vec<DensityPt>>,
    pub density_map: Option<DensityMap>,
    pub density_rect: Option<DensityRect>, // todo: Remove?
    pub aa_seq: Vec<AminoAcid>,
    pub experimental_method: Option<ExperimentalMethod>,
    /// E.g: ["A", "B"]. Inferred from atoms.
    pub alternate_conformations: Option<Vec<String>>,
    /// Index. Ones present are displayed. Used for various UI filers like "near lig only", or "nearby sel only"
    pub atoms_filtered_to_disp: Option<Vec<usize>>,
    /// For color-coding based on SIFTS (From Uniprot/PDBe)
    pub sifts_mapping: Option<Vec<SiftsUniprotMapping>>,
    /// The mmCIF text this peptide was built from, for saving it back out with everything we
    /// don't parse intact. Adding, removing, and detaching ligands (see `peptide_ligands`) keep this
    /// in sync with the peptide.
    pub source_cif: Option<String>,
}

/// The extended, 12-character form of a PDB ID, e.g. `1CRN` -> `"pdb_00001crn"`. Accepts either
/// form. `None` if this isn't a PDB ID.
pub fn pdb_id_extended(id: &str) -> Option<String> {
    let id = id.trim().to_ascii_lowercase();

    // Legacy IDs start with a nonzero digit.
    let legacy = |v: &str| {
        v.len() == 4
            && v.starts_with(|c: char| matches!(c, '1'..='9'))
            && v.chars().all(|c| c.is_ascii_alphanumeric())
    };

    if legacy(&id) {
        return Some(format!("pdb_0000{id}"));
    }

    let body = id.strip_prefix("pdb_")?;
    (body.len() == 8 && body.chars().all(|c| c.is_ascii_alphanumeric())).then_some(id)
}

/// The legacy, 4-character form of a PDB ID, e.g. `pdb_00001crn` -> `"1crn"`. Accepts either
/// form. `None` if this isn't a PDB ID, or is an extended one with no legacy form.
pub fn pdb_id_legacy(id: &str) -> Option<String> {
    pdb_id_extended(id)?
        .strip_prefix("pdb_0000")
        .filter(|v| !v.starts_with('0'))
        .map(str::to_owned)
}

fn push_ident(idents: &mut Vec<PeptideIdent>, ident: PeptideIdent) {
    if !idents.contains(&ident) {
        idents.push(ident);
    }
}

/// Add the RCSB ident for a PDB ID, in its extended form, and PDBe's, which is the same entry.
/// PDBe keys on the legacy form where there is one. Does nothing if this isn't a PDB ID.
fn push_pdb_idents(idents: &mut Vec<PeptideIdent>, id: &str) {
    let Some(extended) = pdb_id_extended(id) else {
        return;
    };
    let pdbe = pdb_id_legacy(&extended).unwrap_or_else(|| extended.clone());

    push_ident(idents, PeptideIdent::Rcsb(extended));
    push_ident(idents, PeptideIdent::Pdbe(pdbe));
}

/// Identifiers from an mmCIF's cross-references: its own IDs (`_database_2`), related entries
/// (`_pdbx_database_related`), and its entities' sequences (`_struct_ref`). If none identify the
/// entry itself, falls back to its entry ID. PDB IDs are converted to their extended form.
pub fn idents_from_mmcif(m: &MmCif) -> Vec<PeptideIdent> {
    let mut result = Vec::new();

    for id in &m.database_ids {
        match id.database.to_ascii_uppercase().as_str() {
            // The code is the legacy PDB ID, and the accession, where present, the extended one.
            "PDB" => {
                push_pdb_idents(&mut result, &id.code);
                if let Some(accession) = &id.accession {
                    push_pdb_idents(&mut result, accession);
                }
            }
            "EMDB" => push_ident(&mut result, PeptideIdent::Emdb(id.code.clone())),
            "BMRB" => push_ident(&mut result, PeptideIdent::Bmrb(id.code.clone())),
            "ALPHAFOLDDB" => push_ident(&mut result, PeptideIdent::AlphaFoldDb(id.code.clone())),
            _ => (),
        }
    }

    // Older entries list their EMDB map and BMRB data here instead of in `_database_2`.
    for entry in &m.related_entries {
        match entry.db_name.to_ascii_uppercase().as_str() {
            // Not e.g. `other EM volume`: maps of other states, which this model isn't built into.
            "EMDB"
                if entry
                    .content_type
                    .as_deref()
                    .is_some_and(|c| c.eq_ignore_ascii_case("associated EM volume")) =>
            {
                push_ident(&mut result, PeptideIdent::Emdb(entry.db_id.clone()))
            }
            "BMRB" => push_ident(&mut result, PeptideIdent::Bmrb(entry.db_id.clone())),
            _ => (),
        }
    }

    for struct_ref in &m.struct_refs {
        if let Some(accession) = &struct_ref.accession
            && matches!(
                struct_ref.db_name.to_ascii_uppercase().as_str(),
                "UNP" | "UNIPROT" | "UNIPROTKB"
            )
        {
            push_ident(&mut result, PeptideIdent::Uniprot(accession.clone()));
        }
    }

    let has_entry_id = result
        .iter()
        .any(|i| matches!(i, PeptideIdent::Rcsb(_) | PeptideIdent::AlphaFoldDb(_)));

    if !has_entry_id {
        let entry_id = m.ident.trim();

        if entry_id.starts_with("AF-") {
            push_ident(&mut result, PeptideIdent::AlphaFoldDb(entry_id.to_owned()));
        } else {
            push_pdb_idents(&mut result, entry_id);
        }
    }

    result
}

impl MoleculePeptide {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        chains: Vec<Chain>,
        residues: Vec<Residue>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let (center, size) = mol_center_size(&atoms);

        let mut result = Self {
            // We create bonds only after
            common: MoleculeCommon::new(ident, atoms, bonds, metadata, path),
            chains,
            residues,
            center,
            size,
            ..Default::default()
        };

        result.aa_seq = result.get_seq();
        result.bonds_hydrogen = create_hydrogen_bonds_single_mol(
            &result.common.atoms,
            &result.common.atom_posits,
            &result.common.bonds,
        );

        // Override the one set in Common::new(), now that we've added hydrogens.
        result.common.build_adjacency_list();

        result.update_het_residues();

        // Ideally, alternate conformations should go here, but we place them in from_mmcif
        // so they can be added prior to Hydrogens.
        result
    }

    /// Refresh `het_residues` from `residues`.
    pub fn update_het_residues(&mut self) {
        self.het_residues = self
            .residues
            .iter()
            .filter(|r| matches!(r.res_type, ResidueType::Other(_)) && r.atoms.len() >= 10)
            .cloned()
            .collect();
    }

    /// If a residue, get the alpha C. If multiple, get an arbitrary one.
    /// todo: Make this work for non-peptides.
    ///
    /// Note: the `Selection`-based wrapper around this lives in Molchanica; selection is a UI concern.
    pub fn get_res_sel_atom(&self, res_i: usize) -> Option<&Atom> {
        let res = self.residues.get(res_i)?;
        if res.atoms.is_empty() {
            return None;
        }

        for atom_i in &res.atoms {
            let atom = &self.common.atoms[*atom_i];
            if let Some(role) = atom.role
                && role == AtomRole::C_Alpha
            {
                return Some(atom);
            }
        }

        // If we can't find  C alpha, default to the first atom.
        Some(&self.common.atoms[res.atoms[0]])
    }

    #[allow(clippy::type_complexity)]
    /// Load RCSB data, and the list of (non-coordinate) files available from the PDB. We do this
    /// in a new thread, to prevent blocking the UI, or delaying a molecule's loading.
    pub fn updates_rcsb_data(
        &mut self,
        pending_data: &mut Option<
            Receiver<(
                Result<PdbDataResults, ReqError>,
                Result<FilesAvailable, ReqError>,
            )>,
        >,
    ) {
        if (self.rcsb_files_avail.is_some() && self.rcsb_data.is_some()) || pending_data.is_some() {
            return;
        }

        let ident = self.common.ident.clone(); // data the worker needs
        let (tx, rx) = mpsc::channel(); // one-shot channel

        println!("Getting RCSB auxiliary data...");

        let start = Instant::now();

        thread::spawn(move || {
            let data = rcsb::get_all_data(&ident);
            let files_data = rcsb::get_files_avail(&ident);

            let elapsed = start.elapsed().as_millis();
            println!("RCSB data loaded in {elapsed:.1}ms");

            let _ = tx.send((data, files_data));
        });

        *pending_data = Some(rx);
    }

    #[allow(clippy::type_complexity)]
    /// Call this periodically from the UI/event loop; it’s non-blocking.
    /// `None` means the worker is still pending. `Some` means it completed, and
    /// the contained flag reports whether molecule data was updated.
    pub fn poll_mol_pending_data(
        &mut self,
        pending_data_avail: &Receiver<(
            Result<PdbDataResults, ReqError>,
            Result<FilesAvailable, ReqError>,
        )>,
    ) -> Option<bool> {
        match pending_data_avail.try_recv() {
            Ok((Ok(pdb_data), Ok(files_avail))) => {
                self.rcsb_data = Some(pdb_data);
                self.rcsb_files_avail = Some(files_avail);
                Some(true)
            }

            // PdbDataResults failed, but FilesAvailable might not have been sent:
            Ok((Err(e), _)) => {
                eprintln!("Failed to fetch PDB data for {}: {e:?}", self.common.ident);
                Some(false)
            }

            // FilesAvailable failed (even if PdbDataResults succeeded):
            Ok((_, Err(e))) => {
                eprintln!("Failed to fetch file‐list for {}: {e:?}", self.common.ident);
                Some(false)
            }

            // The worker hasn’t sent anything yet.
            Err(mpsc::TryRecvError::Empty) => None,

            // The sender hung up before sending.
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Worker thread died before sending result");
                Some(false)
            }
        }
    }
    /// Get the amino acid sequence from the currently opened molecule, if applicable.
    fn get_seq(&self) -> Vec<AminoAcid> {
        // todo: If not a polypeptide, should we return an error, or empty vec?
        let mut result = Vec::new();

        // todo This is fragile, I believe.
        for res in &self.residues {
            if let ResidueType::AminoAcid(aa) = res.res_type {
                result.push(aa);
            }
        }

        result
    }
}

impl MoleculePeptide {
    pub fn from_mmcif(
        mut m: MmCif,
        ff_map: &ProtFfChargeMapSet,
        path: Option<PathBuf>,
        ph: f32,
    ) -> Result<Self, io::Error> {
        // Add hydrogens, FF types, partial charge, and bonds.
        // Sort out alternate conformations prior to adding hydrogens.
        let mut alternate_conformations: Vec<String> = Vec::new();
        for atom in &mut m.atoms {
            if let Some(alt) = &atom.alt_conformation_id
                && !alternate_conformations.contains(alt)
            {
                alternate_conformations.push(alt.to_owned());
            }
        }

        // todo: Handle alternate conformations!
        // todo: For now, we force the first one. This is crude, and ignores alt conformations.
        if !alternate_conformations.is_empty() {
            let mut atoms_ = Vec::new();

            for atom in &m.atoms {
                if let Some(alt) = &atom.alt_conformation_id {
                    if alt == &alternate_conformations[0] {
                        atoms_.push(atom.clone());
                    } else {
                        for res in &mut m.residues {
                            res.atom_sns.retain(|sn| *sn != atom.serial_number);
                        }
                        for chain in &mut m.chains {
                            chain.atom_sns.retain(|sn| *sn != atom.serial_number);
                        }
                    }
                } else {
                    atoms_.push(atom.clone());
                }
            }

            m.atoms = atoms_;
        }
        for a in &m.atoms {
            if !a.hetero && a.serial_number < 200 {
                // println!("A: {a:?}");
            }
        }

        // if !alternate_conformations.is_empty() {
        //     result.alternate_conformations = Some(alternate_conformations);
        // }

        let start = Instant::now();

        let non_hetero_atom_sns: HashSet<u32> = m
            .atoms
            .iter()
            .filter(|atom| !atom.hetero)
            .map(|atom| atom.serial_number)
            .collect();
        let has_non_peptide_polymer = m.residues.iter().any(|residue| {
            matches!(residue.res_type, ResidueType::Other(_))
                && residue
                    .atom_sns
                    .iter()
                    .any(|sn| non_hetero_atom_sns.contains(sn))
        });

        let (bonds_, dihedrals) = if has_non_peptide_polymer {
            println!("Inferring bonds for a mixed polymer structure...");
            // Mixed polymer structures (for example, protein-DNA complexes) cannot be passed
            // through peptide force-field preparation as a single peptide. Preserve every atom
            // and the original complex geometry, and infer display bonds without peptide-only
            // hydrogen, charge, or dihedral assignment.
            (create_bonds(&m.atoms), Vec::new())
        } else {
            println!(
                "Populating protein hydrogens, dihedral angles, FF types and partial charges..."
            );
            prepare_peptide_mmcif(&mut m, ff_map, ph).unwrap_or_else(|e| {
                eprintln!("Error: Unable to prepare a mmCIF file. Maybe it's not a protein? {e:?}");
                // Populate bonds directly in case of an error:
                let bonds = create_bonds(&m.atoms);
                (bonds, Vec::new())
            })
        };

        // todo: Speed this up?
        let end = start.elapsed().as_millis();
        println!("Prepared molecule topology in {end:.1}ms");

        let (atoms, bonds, residues, chains) = molecules::init_bonds_chains_res(
            &m.atoms,
            &bonds_,
            &m.residues,
            &m.chains,
            &dihedrals,
        )?;

        let idents = idents_from_mmcif(&m);

        let mut result = Self::new(
            m.ident.clone(),
            atoms,
            bonds,
            chains,
            residues,
            m.metadata,
            path,
        );

        result.idents = idents;
        result.experimental_method = m.experimental_method;
        result.secondary_structure = m.secondary_structure.clone();

        if !alternate_conformations.is_empty() {
            result.alternate_conformations = Some(alternate_conformations);
        }

        Ok(result)
    }

    /// E.g. run this when pH changes. Removes all hydrogens, and re-adds per the pH. Rebuilds
    /// bonds.
    pub fn reassign_hydrogens(&mut self, ph: f32, ff_map: &ProtFfChargeMapSet) -> io::Result<()> {
        let non_h_sns: HashSet<u32> = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.serial_number)
            .collect();

        let mut atoms_gen = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.to_generic())
            .collect();

        println!("Reassigning H on protein at pH {ph:.1}");

        // Strip old H serial numbers from residues and chains so that
        // populate_hydrogens_dihedrals only appends fresh H SNs. Without
        // this, the stale H SNs remain in atom_sns and Residue::from_generic
        // fails to find them in the (H-filtered) atoms list.
        let mut res_gen: Vec<_> = self
            .residues
            .iter()
            .map(|r| {
                let mut rg = r.to_generic();
                rg.atom_sns.retain(|sn| non_h_sns.contains(sn));
                rg
            })
            .collect();

        let mut chains_gen: Vec<_> = self
            .chains
            .iter()
            .map(|c| {
                let mut cg = c.to_generic();
                cg.atom_sns.retain(|sn| non_h_sns.contains(sn));
                cg
            })
            .collect();

        println!("Populating Hydrogens and dihedral angles...");
        let start = Instant::now();
        // Note: These don't change here, but htis function populates them anyway, so why not.
        let dihedrals =
            populate_hydrogens_dihedrals(&mut atoms_gen, &mut res_gen, &mut chains_gen, ff_map, ph)
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.descrip))?;

        let bonds_gen = create_bonds(&atoms_gen);

        let (atoms, bonds, residues, chains) = molecules::init_bonds_chains_res(
            &atoms_gen,
            &bonds_gen,
            &res_gen,
            &chains_gen,
            &dihedrals,
        )?;

        self.common.atoms = atoms;
        self.common.bonds = bonds;
        self.residues = residues;
        self.chains = chains;

        self.common.build_adjacency_list();
        self.common.reset_posits();

        // Hydrogen reassignment infers bonds again; restore the source component topology.
        if let Some(text) = &self.source_cif {
            let doc = crate::mmcif_edit::CifDoc::new(text)?;
            self.apply_component_bonds(&doc);
        }

        let elapsed = start.elapsed().as_millis();
        let h_count = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element == Element::Hydrogen)
            .count();

        println!("{h_count} Hydrogens populated in {elapsed:.1} ms");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdb_ids_convert_between_forms() {
        assert_eq!(pdb_id_extended("1CRN").as_deref(), Some("pdb_00001crn"));
        assert_eq!(
            pdb_id_extended(" pdb_00001CRN").as_deref(),
            Some("pdb_00001crn")
        );
        assert_eq!(
            pdb_id_extended("pdb_10000abc").as_deref(),
            Some("pdb_10000abc")
        );
        assert_eq!(pdb_id_extended("0ABC"), None);
        assert_eq!(pdb_id_extended("CRN"), None);
        assert_eq!(pdb_id_extended("AF-P69905-F1"), None);

        assert_eq!(pdb_id_legacy("pdb_00001crn").as_deref(), Some("1crn"));
        assert_eq!(pdb_id_legacy("1CRN").as_deref(), Some("1crn"));
        assert_eq!(pdb_id_legacy("pdb_10000abc"), None);
    }

    /// As the RCSB writes them: loops, with text fields, for an entry with several entities.
    /// Abridged from 6GO7, with 6VSB's EMDB rows.
    #[test]
    fn idents_from_rcsb_cif() {
        let text = "data_6GO7
_entry.id   6GO7
#
loop_
_database_2.database_id
_database_2.database_code
_database_2.pdbx_database_accession
_database_2.pdbx_DOI
PDB   6GO7         pdb_00006go7 10.2210/pdb6go7/pdb
WWPDB D_1200010309 ?            ?
EMDB  EMD-21375    ?            ?
#
loop_
_pdbx_database_related.db_name
_pdbx_database_related.details
_pdbx_database_related.db_id
_pdbx_database_related.content_type
EMDB 'Prefusion spike, one RBD up' EMD-21375 'associated EM volume'
EMDB .                             EMD-21374 'other EM volume'
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.db_code
_struct_ref.pdbx_db_accession
_struct_ref.pdbx_db_isoform
_struct_ref.entity_id
_struct_ref.pdbx_seq_one_letter_code
_struct_ref.pdbx_align_begin
1 UNP TDT_MOUSE   P09838 ?        1
;SPSPVPGSQNVPAPAVKKISQYACQRRTTLNNYNQLFTDALDILAENDELRENEGSCLAFMRASSVLKSLPFPITSMKDT
QGLLLY
;
132
2 UNP DPOLM_MOUSE Q9JIW4 ?        1 HQYHRSHLADSAHNLRQRSSTMDAFERSFC 363
3 UNP TDT_MOUSE   P09838 P09838-2 1
;ILKLDHGRVHSEKSGQQEGKGWKAIRVDLVMCPYDRRAFALLGWTGSRQFERDLRRYATHERKMMLDNHALYDRTKRVFL
;
407
4 PDB 6GO7        6GO7   ?        2 ? 1
#
";
        let idents = idents_from_mmcif(&MmCif::new(text).unwrap());

        assert_eq!(
            idents,
            vec![
                PeptideIdent::Rcsb("pdb_00006go7".to_owned()),
                PeptideIdent::Pdbe("6go7".to_owned()),
                PeptideIdent::Emdb("EMD-21375".to_owned()),
                PeptideIdent::Uniprot("P09838".to_owned()),
                PeptideIdent::Uniprot("Q9JIW4".to_owned()),
            ]
        );
    }

    /// As AlphaFold DB writes them: key-value items, with its own database name.
    #[test]
    fn idents_from_alphafold_cif() {
        let text = "data_AF-P69905-F1
#
_entry.id AF-P69905-F1
#
_database_2.database_code AF-P69905-F1
_database_2.database_id   AlphaFoldDB
#
_struct_ref.db_code                  HBA_HUMAN
_struct_ref.db_name                  UNP
_struct_ref.pdbx_db_accession        P69905
_struct_ref.pdbx_db_isoform          ?
_struct_ref.pdbx_seq_one_letter_code
;MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNA
LSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
;
#
";
        assert_eq!(
            idents_from_mmcif(&MmCif::new(text).unwrap()),
            vec![
                PeptideIdent::AlphaFoldDb("AF-P69905-F1".to_owned()),
                PeptideIdent::Uniprot("P69905".to_owned()),
            ]
        );
    }

    /// An older NMR entry, with its BMRB data only listed as related, and identified by its entry
    /// ID alone; and an mmCIF with no identifiers.
    #[test]
    fn idents_from_related_entries_and_entry_id() {
        let text = "data_2K39
_entry.id   2K39
#
_pdbx_database_related.db_name        BMRB
_pdbx_database_related.db_id          15772
_pdbx_database_related.content_type   unspecified
_pdbx_database_related.details        .
#
";
        assert_eq!(
            idents_from_mmcif(&MmCif::new(text).unwrap()),
            vec![
                PeptideIdent::Bmrb("15772".to_owned()),
                PeptideIdent::Rcsb("pdb_00002k39".to_owned()),
                PeptideIdent::Pdbe("2k39".to_owned()),
            ]
        );

        let unidentified = MmCif {
            ident: "MD run".to_owned(),
            ..Default::default()
        };
        assert!(idents_from_mmcif(&unidentified).is_empty());
    }
}
