//! Fundamental data structures for small organic molecules / ligands

use std::{
    collections::{HashMap, HashSet},
    io,
    path::{Path, PathBuf},
    sync::{mpsc, mpsc::Receiver},
    thread,
};

use bio_apis::{
    ReqError, amber_geostd,
    amber_geostd::GeostdData,
    pubchem,
    pubchem::{ProteinStructure, StructureSearchNamespace, properties},
};
use bio_files::{
    ChargeType, Mol2, MolType, Pdbqt, PharmacophoreFeatureGeneric, Sdf, Xyz, create_bonds,
    md_params::{ForceFieldParams, ForceFieldParamsVec},
};
use dynamics::{
    param_inference::{AmberDefSet, assign_missing_params, find_ff_types},
    partial_charge_inference::infer_charge,
};
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::{
    mol_components::MolComponents,
    molecules::{
        Atom, Bond, Chain, MolGeneric, MolGenericRef, MolIdent, MolIdentType,
        PHARMACOPHORE_POCKET_ATOMS_KEY, Residue,
        common::MoleculeCommon,
        conformers::{Conformer, characterize_conformations},
        pocket::Pocket,
    },
    properties::{mol_characterization::MolCharacterization, therapeutic::TherapeuticProperties},
    screening::pharmacophore::{Pharmacophore, PharmacophoreFeature},
};

/// A molecule representing a small organic molecule. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeSmall {
    pub common: MoleculeCommon,
    pub idents: Vec<MolIdent>,
    /// FF type and partial charge on all atoms. Quick lookup flag.
    pub ff_params_loaded: bool,
    /// E.g., overrides for dihedral angles (part of the *bonded* dynamics calculation) for this
    /// specific molecule, as provided by Amber. Quick lookup flag.
    pub frcmod_loaded: bool,
    /// E.g. loaded proteins from Pubchem.
    pub associated_structures: Vec<ProteinStructure>,
    pub characterization: Option<MolCharacterization>,
    pub conformer: Option<Conformer>,
    pub pharmacophore: Pharmacophore,
    pub therapeutic_props: Option<TherapeuticProperties>,
    pub components: Option<MolComponents>,
}

/// Metadata keys used to persist molecule identifiers in formats that have no dedicated place for
/// them (SDF data fields; our `@`-prefixed Mol2 equivalent). The first key of each list is the one
/// we write; the rest are alternates we accept on load, e.g. the tags PubChem, ChEBI and DrugBank
/// use in their own downloads. Matched case-insensitively.
///
/// ChEBI and PDBe are the main motivation: neither database includes its own accession in the
/// files it serves, so once we look one up online, this is how it survives a save and re-load.
const MD_KEYS_PUBCHEM: &[&str] = &[
    "PUBCHEM_COMPOUND_CID",
    // How ChEBI identifies a PubChem CID.
    "PubChem Compound Database Links",
    "PUBCHEM_CID",
];
/// ChEBI writes its accession as `ChEBI ID` in the bulk DB SDFs it distributes, with a `CHEBI:` prefix on
/// the value. (Its single-structure downloads are bare Molfiles with no data fields at all.)
const MD_KEYS_CHEBI: &[&str] = &["ChEBI ID", "CHEBI_ID", "ChEBI Database Links"];
/// PDBe/Amber GeoStd chemical component idents, e.g. "ATP". No source we load from publishes a tag
/// for these, so `PDBE_ID` is ours.
const MD_KEYS_PDBE: &[&str] = &["PDBE_ID", "PDBeChem Database Links", "PDB Database Links"];
const MD_KEYS_DRUGBANK: &[&str] = &["DRUGBANK_ID", "DrugBank Database Links"];
const MD_KEYS_SMILES: &[&str] = &[
    "SMILES",
    "PUBCHEM_SMILES",
    "PUBCHEM_OPENEYE_ISO_SMILES",
    "PUBCHEM_OPENEYE_CAN_SMILES",
];
const MD_KEYS_INCHI: &[&str] = &["INCHI", "PUBCHEM_IUPAC_INCHI"];
const MD_KEYS_INCHI_KEY: &[&str] = &["INCHIKEY", "PUBCHEM_IUPAC_INCHIKEY"];
const MD_KEYS_IUPAC_NAME: &[&str] = &["IUPAC_NAME", "PUBCHEM_IUPAC_NAME"];
const MD_KEYS_PUBCHEM_TITLE: &[&str] = &["PUBCHEM_TITLE"];
/// HMDB's own SDF distribution puts its accession in the generic `DATABASE_ID` field (paired with
/// `DATABASE_NAME`), so `HMDB_ID` is ours; the rest are cross-references other sources publish.
/// ChEBI uses the "HMDB Database Links" metadata tag to indicate these.
const MD_KEYS_HMDB: &[&str] = &["HMDB_ID", "HMDB Database Links", "HMDB"];

/// DrugBank's SDF distribution names its source database instead of using a DrugBank-specific tag.
const MD_KEY_DB_NAME: &str = "DATABASE_NAME";
const MD_KEY_DB_ID: &str = "DATABASE_ID";

// Seen in ChEBI's bulk download SDF. implementation: todo.
const MD_KEYS_KEGG: &[&str] = &["KEGG COMPOUND Database Links"];

/// Case-insensitive metadata lookup over candidate keys, in priority order. Values that list
/// several cross-references, one per line, are reduced to the first.
fn md_get<'a>(metadata: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        for (k, v) in metadata {
            if !k.eq_ignore_ascii_case(key) {
                continue;
            }

            let v = v.lines().next().unwrap_or_default().trim();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }

    None
}

/// ChEBI accessions are conventionally written `CHEBI:15377`; accept a bare number as well.
fn parse_chebi_id(val: &str) -> Option<u32> {
    let val = val.trim();
    let digits = match val.get(..6) {
        Some(prefix) if prefix.eq_ignore_ascii_case("CHEBI:") => &val[6..],
        _ => val,
    };

    digits.trim().parse().ok()
}

/// HMDB accessions are conventionally written `HMDB0000122`: an `HMDB` prefix, then a zero-padded
/// number. (Pre-2019 accessions used five digits instead of seven.) Accept a bare number as well.
fn parse_hmdb_id(val: &str) -> Option<u32> {
    let val = val.trim();
    let digits = match val.get(..4) {
        Some(prefix) if prefix.eq_ignore_ascii_case("HMDB") => &val[4..],
        _ => val,
    };

    digits.trim().parse().ok()
}

/// The conventional way to write an HMDB accession, e.g. `122` -> `"HMDB0000122"`. The inverse of
/// [`parse_hmdb_id`]; HMDB itself always writes the prefix and the padding, so this is the form we
/// store in metadata and show in the UI.
pub fn hmdb_accession(id: u32) -> String {
    format!("HMDB{id:07}")
}

/// Extract identifiers from file metadata. This is the load half of the round trip;
/// `MoleculeSmall::metadata_with_ids_pocket` is the save half.
pub fn idents_from_metadata(ident: &str, metadata: &HashMap<String, String>) -> Vec<MolIdent> {
    let mut result = Vec::new();

    if let Some(v) = md_get(metadata, MD_KEYS_PUBCHEM)
        && let Ok(cid) = v.parse::<u32>()
    {
        result.push(MolIdent::PubChem(cid));
    }

    if let Some(v) = md_get(metadata, MD_KEYS_CHEBI)
        && let Some(id) = parse_chebi_id(v)
    {
        result.push(MolIdent::Chebi(id));
    }

    if let Some(v) = md_get(metadata, MD_KEYS_HMDB)
        && let Some(id) = parse_hmdb_id(v)
    {
        result.push(MolIdent::Hmdb(id));
    }

    if let Some(v) = md_get(metadata, MD_KEYS_PDBE) {
        result.push(MolIdent::PdbeAmber(v.to_owned()));
    }

    if let Some(v) = md_get(metadata, MD_KEYS_DRUGBANK) {
        result.push(MolIdent::DrugBank(v.to_owned()));
    }

    // Seen in ChEBI, and in the tags we write ourselves. Not on PubChem SDFs, which use the
    // `PUBCHEM_`-prefixed alternates.
    if let Some(v) = md_get(metadata, MD_KEYS_SMILES) {
        result.push(MolIdent::Smiles(v.to_owned()));
    }
    if let Some(v) = md_get(metadata, MD_KEYS_INCHI) {
        result.push(MolIdent::InchI(v.to_owned()));
    }
    if let Some(v) = md_get(metadata, MD_KEYS_INCHI_KEY) {
        result.push(MolIdent::InchIKey(v.to_owned()));
    }
    if let Some(v) = md_get(metadata, MD_KEYS_IUPAC_NAME) {
        result.push(MolIdent::IupacName(v.to_owned()));
    }
    if let Some(v) = md_get(metadata, MD_KEYS_PUBCHEM_TITLE) {
        result.push(MolIdent::PubchemTitle(v.to_owned()));
    }

    if let Some(db_name) = md_get(metadata, &[MD_KEY_DB_NAME]) {
        if db_name.eq_ignore_ascii_case("drugbank") {
            if let Some(v) = md_get(metadata, &[MD_KEY_DB_ID]) {
                result.push(MolIdent::DrugBank(v.to_owned()));
            }
            // This seems to be valid for Drugbank-sourced molecules.
            if let Ok(cid) = ident.parse::<u32>() {
                result.push(MolIdent::PubChem(cid));
            }
        }

        // HMDB's own SDF distribution tags its accession this way, in addition to `HMDB_ID`.
        if db_name.eq_ignore_ascii_case("hmdb")
            && let Some(v) = md_get(metadata, &[MD_KEY_DB_ID])
            && let Some(id) = parse_hmdb_id(v)
        {
            result.push(MolIdent::Hmdb(id));
        }
    }

    if !ident.is_empty()
        && ident.len() <= 4
        && ident.parse::<u32>().is_err()
        && !result.iter().any(|i| matches!(i, MolIdent::PdbeAmber(_)))
    {
        // This is a guess
        result.push(MolIdent::PdbeAmber(ident.to_owned()));
    }

    result
}

impl MoleculeSmall {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let mut idents = idents_from_metadata(&ident, &metadata);

        let common = MoleculeCommon::new(ident, atoms, bonds, metadata, path);

        // Fall back to a SMILES string derived from the structure if the file didn't carry one.
        if !idents.iter().any(|i| matches!(i, MolIdent::Smiles(_))) {
            idents.push(MolIdent::Smiles(common.to_smiles()));
        }

        // Sources overlap; e.g. a PubChem CID can arrive under two different tags.
        let mut seen = HashSet::new();
        idents.retain(|ident| seen.insert(ident.clone()));

        Self {
            common,
            idents,
            ..Default::default()
        }
    }

    pub fn update_characterization(&mut self) {
        self.characterization = Some(MolCharacterization::new(&self.common));

        // For now, this works as the spot
        self.components = MolComponents::new(&self);
        self.conformer = None;
    }

    pub fn update_conformer(&mut self, ff_params: &ForceFieldParams) {
        if self.characterization.is_none() {
            self.update_characterization();
        }

        self.conformer = characterize_conformations(self, ff_params);
    }

    /// Returns the first if multiple entries of a given ident type exist for
    /// this molecule.
    pub fn get_ident(&self, ident_type: MolIdentType) -> Option<&MolIdent> {
        for ident in &self.idents {
            if ident.ident_type() == ident_type {
                return Some(ident);
            }
        }

        None
    }

    /// Perhaps redundant with `get_ident`.
    pub fn get_smiles(&self) -> Option<&str> {
        for ident in &self.idents {
            if let MolIdent::Smiles(id) = ident {
                return Some(id);
            }
        }

        None
    }
}

impl MolGeneric for MoleculeSmall {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::Small(self)
    }

    fn mol_type(&self) -> crate::molecules::MolType {
        crate::molecules::MolType::Ligand
    }
}

impl TryFrom<Mol2> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Mol2) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Note: We don't compute bonds here; we assume they're included in the molecule format.
        // Handle path after; not supported by TryFrom.

        let mut res = Self::new(m.ident, atoms, bonds, m.metadata.clone(), None);

        res.pharmacophore = pharmacophore_from_biofiles(
            &m.pharmacophore_features,
            &m.metadata,
            &res.common.atoms,
            &res.common.ident,
        )?;

        res.common.metadata.remove(PHARMACOPHORE_POCKET_ATOMS_KEY);

        Ok(res)
    }
}

impl TryFrom<Sdf> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Sdf) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Handle path and state-specific items after; not supported by TryFrom.
        let mut res = Self::new(m.ident, atoms, bonds, m.metadata.clone(), None);

        res.pharmacophore = pharmacophore_from_biofiles(
            &m.pharmacophore_features,
            &m.metadata,
            &res.common.atoms,
            &res.common.ident,
        )?;

        Ok(res)
    }
}

impl MoleculeSmall {
    pub fn from_xyz(m: Xyz, path: &Path) -> io::Result<Self> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds_gen = create_bonds(&m.atoms);
        let bonds: Vec<Bond> = bonds_gen
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert(String::from("Comment"), m.comment.clone());

        // Handle path and state-specific items after; not supported by TryFrom.
        Ok(Self::new(
            filename,
            atoms,
            bonds,
            metadata,
            Some(path.to_owned()),
        ))
    }
}

impl TryFrom<Pdbqt> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Pdbqt) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();
        let mut residues = Vec::with_capacity(m.residues.len());
        for res in &m.residues {
            residues.push(Residue::from_generic(res, &atoms)?);
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Handle path after; not supported by TryFrom.
        Ok(Self::new(
            m.ident,
            atoms,
            bonds,
            HashMap::new(), // todo: Metadata?
            None,
        ))
    }
}

impl MoleculeSmall {
    /// Augment this molecule's metadata with IDs; run this prior to saving. This ensures these are
    /// saved and loaded in file formats, as our internal fields don't map directly to these. (Mol2, SDF etc)
    ///
    /// Also, serialize the pocket atoms.
    fn metadata_with_ids_pocket(&self) -> HashMap<String, String> {
        let mut res = self.common.metadata.clone();

        // Note: If already present, these may be redundant with metadata already loaded.
        // Insert them here in case they're not.

        for ident in &self.idents {
            match ident {
                MolIdent::PubChem(cid) => {
                    res.insert(MD_KEYS_PUBCHEM[0].to_string(), cid.to_string());
                }
                MolIdent::Chebi(id) => {
                    res.insert(MD_KEYS_CHEBI[0].to_string(), format!("CHEBI:{id}"));
                }
                MolIdent::PdbeAmber(id) => {
                    res.insert(MD_KEYS_PDBE[0].to_string(), id.clone());
                }
                MolIdent::DrugBank(id) => {
                    res.insert(MD_KEYS_DRUGBANK[0].to_string(), id.clone());
                    // The pair DrugBank's own SDFs use.
                    res.insert(MD_KEY_DB_ID.to_string(), id.clone());
                    res.insert(MD_KEY_DB_NAME.to_string(), "drugbank".to_string());
                }
                MolIdent::Smiles(v) => {
                    res.insert(MD_KEYS_SMILES[0].to_string(), v.clone());
                }
                MolIdent::InchI(v) => {
                    res.insert(MD_KEYS_INCHI[0].to_string(), v.clone());
                }
                MolIdent::InchIKey(v) => {
                    res.insert(MD_KEYS_INCHI_KEY[0].to_string(), v.clone());
                }
                MolIdent::IupacName(v) => {
                    res.insert(MD_KEYS_IUPAC_NAME[0].to_string(), v.clone());
                }
                MolIdent::PubchemTitle(v) => {
                    res.insert(MD_KEYS_PUBCHEM_TITLE[0].to_string(), v.clone());
                }
                MolIdent::Hmdb(id) => {
                    res.insert(MD_KEYS_HMDB[0].to_string(), hmdb_accession(*id));
                }
                MolIdent::Kegg(v) => {
                    res.insert(MD_KEYS_KEGG[0].to_string(), v.clone());
                }
            }
        }

        // Save the atoms in the pocket, for reconstruction upon load.
        if let Some(pocket) = &self.pharmacophore.pocket {
            let mut md_val = String::new();

            for atom in &pocket.common.atoms {
                md_val.push_str(&format!(
                    "{}    {}    {:.5}    {:.5}    {:.5}\n",
                    atom.serial_number,
                    atom.element.to_letter(),
                    atom.posit.x,
                    atom.posit.y,
                    atom.posit.z
                ));
            }

            res.insert(PHARMACOPHORE_POCKET_ATOMS_KEY.to_string(), md_val);
        }

        res
    }

    pub fn to_sdf(&self) -> Sdf {
        // SDF doesn't support explicit atom SNs; they use order. This reassignment makes sure
        // the bond atom assignments aren't lost in this process.
        let (atoms, bonds) = {
            let mut common_reassigned = self.common.clone();
            common_reassigned.reassign_sns();

            let a = common_reassigned
                .atoms
                .iter()
                .map(|a| a.to_generic())
                .collect();
            let b = common_reassigned
                .bonds
                .iter()
                .map(|b| b.to_generic())
                .collect();

            (a, b)
        };

        Sdf {
            ident: self.common.ident.clone(),
            metadata: self.metadata_with_ids_pocket(),
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
            pharmacophore_features: pharmacophore_to_biofiles(&self.pharmacophore)
                .unwrap_or_default(),
        }
    }

    pub fn to_mol2(&self) -> Mol2 {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Mol2 {
            ident: self.common.ident.clone(),
            atoms,
            bonds,
            metadata: self.metadata_with_ids_pocket(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            pharmacophore_features: pharmacophore_to_biofiles(&self.pharmacophore)
                .unwrap_or_default(),
            comment: None,
        }
    }

    pub fn to_xyz(&self) -> Xyz {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();

        let comment = match self.common.metadata.get("Comment") {
            Some(v) => v.to_owned(),
            None => String::new(),
        };

        Xyz { atoms, comment }
    }

    pub fn to_pdbqt(&self) -> Pdbqt {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Pdbqt {
            ident: self.common.ident.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            comment: None,
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
        }
    }
}

impl MoleculeSmall {
    /// For example, this can be used to create a ligand from a residue that was loaded with a mmCIF
    /// file from RCSB. It can then be used for docking, or saving to a Mol2 or SDF file.
    ///
    /// `atoms` here should be the full set, as indexed by `res`, unless `use_sns` is true.
    /// `use_sns` = false is faster.
    ///
    /// We assume the residue is already populated with hydrogens.
    ///
    /// We reposition its atoms to be around the origin.
    pub fn from_res(res: &Residue, atoms: &[Atom], bonds: &[Bond]) -> Self {
        let mut atoms_this = Vec::with_capacity(res.atoms.len());

        // We use this map when rebuilding bonds.
        // Old index: (new index, new sn)
        let mut bond_map = HashMap::new();

        for (i, &atom_i_orig) in res.atoms.iter().enumerate() {
            let atom = &atoms[atom_i_orig];

            let serial_number = i as u32 + 1;
            bond_map.insert(atom_i_orig, (i, serial_number));

            atoms_this.push(Atom {
                serial_number,
                residue: None,
                chain: None,
                ..atom.clone()
            });
        }

        let atom_orig_i: Vec<_> = bond_map.keys().collect();
        let bonds_this: Vec<_> = bonds
            .iter()
            .filter(|b| atom_orig_i.contains(&&b.atom_0) && atom_orig_i.contains(&&b.atom_1))
            .cloned()
            .collect();

        let mut bonds_new = Vec::with_capacity(bonds_this.len());
        for bond in &bonds_this {
            let (atom_0, atom_0_sn) = bond_map.get(&bond.atom_0).unwrap();
            let (atom_1, atom_1_sn) = bond_map.get(&bond.atom_1).unwrap();

            bonds_new.push(Bond {
                bond_type: bond.bond_type,
                atom_0_sn: *atom_0_sn,
                atom_1_sn: *atom_1_sn,
                atom_0: *atom_0,
                atom_1: *atom_1,
                is_backbone: false,
            })
        }

        let name = res.res_type.to_string();
        let mut result = Self::new(name.clone(), atoms_this, bonds_new, HashMap::new(), None);

        result.common.center_local_posits_around_origin();

        result.idents.push(MolIdent::PdbeAmber(name));

        result
    }

    pub fn apply_geostd_data(
        &mut self,
        data: GeostdData,
        lig_specific: &mut HashMap<String, ForceFieldParams>,
    ) {
        if !self.ff_params_loaded {
            let Ok(mol2) = Mol2::new(&data.mol2) else {
                eprintln!("Error: No Mol2 available from Geostd");
                return;
            };

            let mut count_c_orig: u32 = 0;
            let mut count_n_orig: u32 = 0;
            let mut count_o_orig: u32 = 0;
            let mut count_h_orig: u32 = 0;
            //
            let mut count_c_amber: u32 = 0;
            let mut count_n_amber: u32 = 0;
            let mut count_o_amber: u32 = 0;
            let mut count_h_amber: u32 = 0;

            for atom in &self.common.atoms {
                match atom.element {
                    Element::Carbon => count_c_orig += 1,
                    Element::Nitrogen => count_n_orig += 1,
                    Element::Oxygen => count_o_orig += 1,
                    Element::Hydrogen => count_h_orig += 1,
                    _ => {}
                }
            }
            for atom in &mol2.atoms {
                match atom.element {
                    Element::Carbon => count_c_amber += 1,
                    Element::Nitrogen => count_n_amber += 1,
                    Element::Oxygen => count_o_amber += 1,
                    Element::Hydrogen => count_h_amber += 1,
                    _ => {}
                }
            }

            if count_c_orig != count_c_amber
                || count_n_orig != count_n_amber
                || count_o_orig != count_o_amber
                || count_h_orig != count_h_amber
            {
                eprintln!(
                    "Unable to load Amber Geostd data for this molecule; atom count mismatch."
                );
                return;
            }

            let mol: Self = match mol2.try_into() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Problem loading Mol2 from geostd: {e}");
                    return; // OK only if this fn returns ()
                }
            };

            self.common.atoms = mol.common.atoms;
            self.common.bonds = mol.common.bonds;
            self.common.atom_posits = mol.common.atom_posits;
            self.common.adjacency_list = mol.common.adjacency_list;

            self.ff_params_loaded = true;
            println!("Loaded Amber Geostd FF data for {}", self.common.ident);
        }

        if !self.frcmod_loaded
            && let Some(f) = data.frcmod
            && let Ok(frcmod) = ForceFieldParamsVec::from_frcmod(&f)
        {
            lig_specific.insert(self.common.ident.clone(), ForceFieldParams::new(&frcmod));
            self.frcmod_loaded = true;

            println!("Loaded Amber FRCMOD data for {}", self.common.ident);
        }
    }

    /// Attempt to find FF type, partial charge, and FRCMOD overrides for a given molecule.
    /// Launch this in a thread.
    ///
    /// Unfortunately, we can't directly map atoms from our original molecule to
    /// the Geostd one. We could do this with coordinates, but that might be complicated.
    /// For now, we perform a sanity check about atom count by element. If it passes,
    /// we replace molecule atom and bond data with that loaded from the mol2.
    fn _search_geostd(
        &mut self,
        ident: &str,
        geostd_thread: &mut Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
        mol_i: usize,
    ) {
        println!("Attempting to load Amber Geostd dynamics data for this molecule...");

        let (tx, rx) = mpsc::channel(); // one-shot channel
        let ident_for_thread = ident.to_string();

        thread::spawn(move || {
            let data = amber_geostd::load_mol_files(&ident_for_thread);
            let _ = tx.send((mol_i, data));
        });

        *geostd_thread = Some(rx);
    }

    /// Refresh the molecule's derived data, and kick off a PubChem properties fetch if we don't
    /// already hold them locally.
    ///
    /// Note: ADME/Tox inference is *not* run here. It lives in the `adme` crate; Molchanica spawns
    /// it alongside this call and stores the result in `therapeutic_props`.
    pub fn update_aux(
        &mut self,
        pubchem_properties_map: &HashMap<MolIdent, pubchem::Properties>,
        pubchem_properties_avail: &mut Option<
            Receiver<(MolIdent, Result<pubchem::Properties, ReqError>)>,
        >,
        ff_params: &ForceFieldParams,
    ) {
        self.update_characterization();
        self.update_conformer(ff_params);

        // Load PubChem properties from either our prefs file, or online. If online,
        // launch this in a separate thread.
        let mut pubchem_ident_exists = false;

        for ident in &self.idents {
            match pubchem_properties_map.get(ident) {
                Some(props) => {
                    println!("Loaded Properties for {ident:?} from our local DB.");

                    self.update_idents_and_char_from_pubchem(props);
                    break;
                }
                None => {
                    let (tx, rx) = mpsc::channel(); // one-shot channel
                    let ident_for_thread = ident.clone();

                    if let MolIdent::PubChem(_) = ident {
                        println!("\nLoading PubChem properties for {ident:?} over HTTP...");

                        thread::spawn(move || {
                            // Part of our borrow-checker workaround
                            let cid: u32 = ident_for_thread.ident_inner().parse().unwrap();
                            let data = pubchem::properties(
                                StructureSearchNamespace::Cid,
                                &cid.to_string(),
                            );

                            let _ = tx.send((ident_for_thread, data));
                        });

                        pubchem_ident_exists = true;
                        *pubchem_properties_avail = Some(rx);
                        break;
                    }
                }
            }
        }

        // If we don't have a PubChemID, use SMILES if we have that. If we have a PDBe/Amber ID,
        // use that to load SMILES. Once we have SMILES, use that to get a PubChem ID.
        if !pubchem_ident_exists {
            for ident in &self.idents {
                let (tx, rx) = mpsc::channel(); // one-shot channel
                let ident_for_thread = ident.clone();

                if let MolIdent::PdbeAmber(_) = ident {
                    println!("\nLoading PubChem properties for {ident:?} over HTTP...");
                    thread::spawn(move || {
                        let data =
                            pubchem::properties_from_pdbe_id(&ident_for_thread.ident_inner());

                        let _ = tx.send((ident_for_thread, data));
                    });

                    *pubchem_properties_avail = Some(rx);
                    break;
                }

                if let MolIdent::Smiles(_) = ident {
                    println!("\nLoading PubChem properties for {ident:?} over HTTP...");
                    thread::spawn(move || {
                        let data = properties(
                            StructureSearchNamespace::Smiles,
                            &ident_for_thread.ident_inner(),
                        );

                        let _ = tx.send((ident_for_thread, data));
                    });

                    *pubchem_properties_avail = Some(rx);
                    break;
                }
            }
        }
    }

    pub fn update_idents_and_char_from_pubchem(&mut self, props: &pubchem::Properties) {
        let mut pubchem_exists = false;
        let mut smiles_exists = false;
        let mut inchi_exists = false;
        let mut inchi_key_exists = false;
        let mut iupac_name_exists = false;
        let mut title_exists = false;

        for ident in &self.idents {
            if matches!(ident, MolIdent::PubChem(_)) {
                pubchem_exists = true;
            }
            if matches!(ident, MolIdent::Smiles(_)) {
                smiles_exists = true;
            }
            if matches!(ident, MolIdent::InchI(_)) {
                inchi_exists = true;
            }
            if matches!(ident, MolIdent::InchIKey(_)) {
                inchi_key_exists = true;
            }
            if matches!(ident, MolIdent::IupacName(_)) {
                iupac_name_exists = true;
            }
            if matches!(ident, MolIdent::PubchemTitle(_)) {
                title_exists = true;
            }
        }

        if !pubchem_exists {
            self.idents.push(MolIdent::PubChem(props.cid));
        }
        if !smiles_exists {
            self.idents.push(MolIdent::Smiles(props.smiles.clone()));
        }
        if !inchi_exists {
            self.idents.push(MolIdent::InchI(props.inchi.clone()));
        }
        if !inchi_key_exists {
            self.idents
                .push(MolIdent::InchIKey(props.inchi_key.clone()));
        }
        if !iupac_name_exists {
            self.idents
                .push(MolIdent::IupacName(props.iupac_name.clone()));
        }
        if !title_exists {
            self.idents
                .push(MolIdent::PubchemTitle(props.title.clone()));
        }

        if let Some(char) = &mut self.characterization {
            char.tpsa_ertl = props.total_polar_surface_area;
            char.volume_pubchem = Some(props.volume);
            char.complexity = Some(props.complexity);
        }
    }

    /// Update partial charges, FF types, and mol-specific params.
    /// Note: Perhaps we restructure? Not all of these need access to state.
    ///
    /// We currently skip mol-specific params for ML training, where we need FF type
    /// and partial charge, but not them.
    pub fn update_ff_related(
        &mut self,
        mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
        gaff2: &ForceFieldParams,
        skip_mol_specific: bool,
    ) {
        self.conformer = None;
        self.ff_params_loaded = true;
        for atom in &self.common.atoms {
            if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                self.ff_params_loaded = false;
                break;
            }
        }

        if mol_specific_param_set
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&self.common.ident))
        {
            self.frcmod_loaded = true;
        }

        // println!("Inferring FF parameter data...");
        // Note: There is an all-in-one `update_small_mol_params` fn we can use as well; it's
        // easier to use nominally, but this approach works better for our this-project Atom and bond types,
        // and loaded flags.

        let mut atoms_gen: Vec<_> = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = self.common.bonds.iter().map(|a| a.to_generic()).collect();

        if !self.ff_params_loaded {
            let defs = AmberDefSet::new().unwrap();
            let ff_types = find_ff_types(&atoms_gen, &bonds_gen, &defs);

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.force_field_type = Some(ff_types[i].clone());

                // We re-use `atoms_gen` for mol specific params below; update atoms gen here.
                atoms_gen[i].force_field_type = Some(ff_types[i].clone());
            }

            let charge = match infer_charge(&atoms_gen, &bonds_gen) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error inferring params: {e:?}");
                    return;
                }
            };

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.partial_charge = Some(charge[i]);
            }

            // // todo: This print and loop are temp.
            // println!("\n FF types computed:");
            // for atom in &self.common.atoms {
            //     println!(
            //         "--{}: {} {:.4}",
            //         atom.serial_number,
            //         atom.force_field_type.as_ref().unwrap(),
            //         atom.partial_charge.unwrap()
            //     );
            // }

            self.ff_params_loaded = true;
        }

        if !self.frcmod_loaded && !skip_mol_specific {
            let mol_specific_params =
                match assign_missing_params(&atoms_gen, &self.common.adjacency_list, gaff2) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "Error inferring params for mol {}: {e:?}",
                            self.common.ident
                        );
                        return;
                    }
                };

            // println!("\n\nDihe FRCMOD created:");
            // for p in &mol_specific_params.dihedral {
            //     println!("\nDihe: {:?}", p);
            // }

            // println!("\n\nImproper FRCMOD created:");
            // for p in &mol_specific_params.improper {
            //     println!("Improp: {:?}", p);
            // }

            mol_specific_param_set.insert(self.common.ident.to_owned(), mol_specific_params);
            self.frcmod_loaded = true;
        }
        // println!("Inference complete.");
    }
}

/// Convert the bio_files SDF or Mol2 metadata-based Pharmacophore layout to our own.
fn pharmacophore_from_biofiles(
    feats: &[PharmacophoreFeatureGeneric],
    metadata: &HashMap<String, String>,
    atoms: &[Atom],
    ident: &str,
) -> io::Result<Pharmacophore> {
    let def = PharmacophoreFeature::default(); // For default vals.

    let mut features = Vec::with_capacity(feats.len());

    for feat in feats {
        // Average position, if multiple atoms.
        let mut posit = Vec3::new_zero();
        let mut atom_i = Vec::with_capacity(feat.atom_sns.len());

        for a in feat.atom_sns.iter() {
            let i = *a as usize - 1;
            if i >= atoms.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Pharmacophore index out of bounds",
                ));
            }

            posit += atoms[i].posit;
            atom_i.push(i);
        }
        posit /= feat.atom_sns.len() as f64;

        features.push(PharmacophoreFeature {
            feature_type: feat.type_.clone().into(),
            posit,
            atom_i,
            ..def.clone()
        });
    }

    // Reconstruct the pocket from serialized atom positions in metadata.
    let pocket = if let Some(atoms_str) = metadata.get(PHARMACOPHORE_POCKET_ATOMS_KEY) {
        let mut pocket_atoms: Vec<Atom> = Vec::new();

        for line in atoms_str.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 {
                continue;
            }

            let Ok(sn) = parts[0].parse::<u32>() else {
                eprintln!("Bad serial number in pocket atom line: {line}");
                continue;
            };
            let Ok(element) = Element::from_letter(parts[1]) else {
                eprintln!("Unknown element in pocket atom line: {line}");
                continue;
            };
            let (Ok(x), Ok(y), Ok(z)) = (
                parts[2].parse::<f64>(),
                parts[3].parse::<f64>(),
                parts[4].parse::<f64>(),
            ) else {
                eprintln!("Bad coordinates in pocket atom line: {line}");
                continue;
            };

            pocket_atoms.push(Atom {
                serial_number: sn,
                posit: Vec3::new(x, y, z),
                element,
                ..Default::default()
            });
        }

        if pocket_atoms.is_empty() {
            None
        } else {
            let common = MoleculeCommon::new(
                format!("{ident}_pocket"),
                pocket_atoms,
                Vec::new(),
                HashMap::new(),
                None,
            );
            Some(Pocket::from(common))
        }
    } else {
        None
    };

    Ok(Pharmacophore {
        name: ident.to_string(),
        mol_ident: ident.to_string(),
        features,
        pocket,
        ..Default::default()
    })
}

fn pharmacophore_to_biofiles(ph: &Pharmacophore) -> io::Result<Vec<PharmacophoreFeatureGeneric>> {
    let mut result = Vec::new();

    for feat in &ph.features {
        if feat.atom_i.is_empty() {
            eprintln!("Pharmacophore feature missing atom index");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Pharmacophore feature missing atom index",
            ));
        };

        let atom_sns = feat.atom_i.iter().map(|i| *i as u32 + 1).collect();
        result.push(PharmacophoreFeatureGeneric {
            atom_sns,
            type_: feat.feature_type.to_generic(),
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use bio_files::{BondType, SdfFormat};
    use lin_alg::f64::Vec3;
    use na_seq::Element;

    use super::*;

    fn temp_path(extension: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mol-defs-idents-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    /// A stand-in molecule with no identifiers in its metadata. Its ident is deliberately longer
    /// than a PDBe chemical component code, so `new` doesn't guess one.
    fn test_mol() -> MoleculeSmall {
        let atoms = vec![
            Atom {
                serial_number: 1,
                posit: Vec3::new_zero(),
                element: Element::Carbon,
                ..Default::default()
            },
            Atom {
                serial_number: 2,
                posit: Vec3::new(1.4, 0., 0.),
                element: Element::Oxygen,
                ..Default::default()
            },
        ];

        let bonds = vec![Bond {
            bond_type: BondType::Single,
            atom_0_sn: 1,
            atom_1_sn: 2,
            atom_0: 0,
            atom_1: 1,
            is_backbone: false,
        }];

        MoleculeSmall::new("Methanol".to_owned(), atoms, bonds, HashMap::new(), None)
    }

    /// ChEBI and PDBe accessions are absent from the files those databases serve, so they only
    /// persist if we write and read our own metadata tags for them.
    #[test]
    fn idents_round_trip_through_sdf_and_mol2() {
        let mut mol = test_mol();

        mol.idents.push(MolIdent::Chebi(15377));
        mol.idents.push(MolIdent::PdbeAmber("ATP".to_owned()));
        mol.idents.push(MolIdent::PubChem(962));
        mol.idents.push(MolIdent::DrugBank("DB09145".to_owned()));
        mol.idents
            .push(MolIdent::InchI("InChI=1S/H2O/h1H2".to_owned()));
        mol.idents
            .push(MolIdent::InchIKey("XLYOFNOQVPJJNP-UHFFFAOYSA-N".to_owned()));
        mol.idents.push(MolIdent::IupacName("oxidane".to_owned()));
        mol.idents.push(MolIdent::PubchemTitle("Water".to_owned()));
        mol.idents.push(MolIdent::Hmdb(2111));

        let expected = mol.idents.clone();

        let sdf_path = temp_path("sdf");
        mol.to_sdf().save(&sdf_path, SdfFormat::V2000).unwrap();
        let from_sdf: MoleculeSmall = Sdf::load(&sdf_path).unwrap().try_into().unwrap();
        fs::remove_file(&sdf_path).unwrap();

        let mol2_path = temp_path("mol2");
        mol.to_mol2().save(&mol2_path).unwrap();
        let from_mol2: MoleculeSmall = Mol2::load(&mol2_path).unwrap().try_into().unwrap();
        fs::remove_file(&mol2_path).unwrap();

        for ident in &expected {
            assert!(
                from_sdf.idents.contains(ident),
                "SDF round trip lost {ident:?}"
            );
            assert!(
                from_mol2.idents.contains(ident),
                "Mol2 round trip lost {ident:?}"
            );
        }
    }

    /// ChEBI's own SDF tag, and the bare-number form.
    #[test]
    fn chebi_accessions_parse_with_and_without_their_prefix() {
        assert_eq!(parse_chebi_id("CHEBI:15377"), Some(15377));
        assert_eq!(parse_chebi_id(" chebi:15377 "), Some(15377));
        assert_eq!(parse_chebi_id("15377"), Some(15377));
        assert_eq!(parse_chebi_id("CHEBI:"), None);

        let mut metadata = HashMap::new();
        metadata.insert("ChEBI ID".to_owned(), "CHEBI:15377".to_owned());
        metadata.insert("PDBE_ID".to_owned(), "HOH".to_owned());

        let idents = idents_from_metadata("water", &metadata);
        assert!(idents.contains(&MolIdent::Chebi(15377)));
        assert!(idents.contains(&MolIdent::PdbeAmber("HOH".to_owned())));
        // The explicit tag wins over the guess made from the molecule's ident.
        assert_eq!(
            idents
                .iter()
                .filter(|i| matches!(i, MolIdent::PdbeAmber(_)))
                .count(),
            1
        );
    }

    /// HMDB's zero-padded accession, the shorter pre-2019 form, and a bare number.
    #[test]
    fn hmdb_accessions_parse_with_and_without_their_prefix() {
        assert_eq!(parse_hmdb_id("HMDB0002111"), Some(2111));
        assert_eq!(parse_hmdb_id(" hmdb00122 "), Some(122));
        assert_eq!(parse_hmdb_id("2111"), Some(2111));
        assert_eq!(parse_hmdb_id("HMDB"), None);

        assert_eq!(hmdb_accession(2111), "HMDB0002111");

        // The tag HMDB writes in the SDFs it distributes, alongside the generic pair.
        let mut metadata = HashMap::new();
        metadata.insert("HMDB_ID".to_owned(), "HMDB0002111".to_owned());

        assert!(idents_from_metadata("", &metadata).contains(&MolIdent::Hmdb(2111)));

        // `DATABASE_ID` alone, as HMDB's own distribution names it.
        let mut metadata = HashMap::new();
        metadata.insert("DATABASE_NAME".to_owned(), "hmdb".to_owned());
        metadata.insert("DATABASE_ID".to_owned(), "HMDB0002111".to_owned());

        assert!(idents_from_metadata("", &metadata).contains(&MolIdent::Hmdb(2111)));
    }
}
