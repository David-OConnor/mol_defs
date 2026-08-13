//! Estimates of how a molecule, in drug form, acts in the human body.
//!
//! These are the outputs of ADME/Tox inference. The types live here so a molecule can carry its
//! predicted properties without depending on the inference library; the inference itself lives in
//! the `adme` crate, which populates these.

use crate::molecules::small::MoleculeSmall;

/// Absorption, distribution, metabolism, and excretion (ADME) properties.
/// I believe this is broadly synonymous with Pharmacokinetics.
///
/// Ones marked "Binary" have training target data of either 0 or 1. We store as
/// floating point, for now, to assist with checking [confidence?].
#[derive(Clone, Debug, Default)]
pub struct Adme {
    // Absorption
    /// TDC.Caco2_Wang. cm/s
    pub intestinal_permeability: f32,
    /// TDC.HIA_Hou. Binary.
    pub intestinal_absorption: f32,
    /// TDC.Pgp_Broccatelli. Binary.
    pub pgp: f32,
    /// Bioavailability_Ma. Binary.
    pub oral_bioavailablity: f32,
    /// TDC.Lipophilicity_AstraZeneca. log-ratio. LogD at pH 7.4.
    pub lipophilicity: f32,
    /// AqSolDB, or TDC.Solubility_AqSolDB. log mol/L
    /// LogS, where S is the aqueous solubility.
    pub solubility_water: f32,
    /// TDC.PAMPA_NCATS
    /// PAMPA (parallel artificial membrane permeability assay) is a commonly employed assay
    /// to evaluate drug permeability across the cellular membrane. Binary.
    pub membrane_permeability: f32,
    /// TDC.hHydrationFreeEnergy_FreeSolv
    /// The Free Solvation Database, FreeSolv(SAMPL), provides experimental and calculated hydration
    /// free energy of small molecules in water. The calculated values are derived from alchemical
    /// free energy calculations using molecular dynamics simulations. todo: Units
    pub hydration_free_energy: f32,
    // Distribution
    /// TDC.BBB_Martins. Binary
    pub blood_brain_barrier: f32,
    /// TDC.PPBR_AZ. % binding value.
    pub plasma_protein_binding_rate: f32,
    /// Volume of Distribution at steady state.
    pub vdss: f32,
    // Metabolism
    /// CYP P450 2C19 Inhibition.
    ///  The CYP P450 genes are essential in the breakdown (metabolism) of various molecules and
    /// chemicals within cells. A drug that can inhibit these enzymes would mean poor metabolism
    /// to this drug and other drugs, which could lead to drug-drug interactions and adverse effects.
    ///
    /// CYP2C19 gene provides instructions for making an enzyme called the endoplasmic reticulum,
    /// which is involved in protein processing and transport.
    /// Binary.
    pub cyp_2c19_inhibition: f32,
    /// CYP2D6 is primarily expressed in the liver. Binary.
    pub cyp_2d6_inhibition: f32,
    pub cyp_3a4_inhibition: f32,
    pub cyp_1a2_inhibition: f32,
    pub cyp_2c9_inhibition: f32,
    // todo: More P450 inhibitions
    // Excretion.
    /// TDC.Half_Life_Obach. Todo: Units.
    pub half_life: f32,
    /// TDC.Clearance_Hepatocyte_AZ. todo: Units.
    pub clearance: f32,
}
#[derive(Clone, Debug, Default)]
pub struct Toxicity {
    /// TDC.LD50_Zhu. log(1/(mol/kg)).
    pub ld50: f32,
    /// TDC.hERG. Related to coordination of the heart's beating. Binary.
    pub ether_a_go_go: f32,
    /// TDC.AMES. Binary.
    pub mutagencity: f32,
    /// TDC.DILI. Binary.
    pub drug_induced_liver_injury: f32,
    /// TDC.Skin_Reaction. Binary.
    pub skin_reaction: f32,
    /// TDC.Carcinogens_lagunin. Binary.
    pub carcinogen: f32,
}

/// Estimates of how the molecule, in drug form, acts in the human body.
/// https://en.wikipedia.org/wiki/Pharmacokinetics
#[derive(Clone, Debug, Default)]
pub struct TherapeuticProperties {
    pub adme: Adme,
    pub toxicity: Toxicity,
    pub breakdown_products: Vec<MoleculeSmall>,
}
