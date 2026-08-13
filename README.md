# Molecule definitions

[![Crate](https://img.shields.io/crates/v/mol_defs.svg)](https://crates.io/crates/mol_defs)
[![Docs](https://docs.rs/mol_defs/badge.svg)](https://docs.rs/mol_defs)

[Home page](https://www.athanorlab.com/rust-tools)

This library contains the molecule data structures used by [Molchanica](https://github.com/David-OConnor/molchanica)
and its ADME inference library. [bio_files](https://crates.io/crates/bio_files) provides simpler
types for reading and writing molecular files; this one contains detailed ones with application-specific data.

Its fundamental types are `Atom`, `Bond`, `Residue`, and `Chain`. Built atop these are the molecule types:
`MoleculeSmall` for small organics, `MoleculePeptide` for proteins, plus `MoleculeNucleicAcid`,
`MoleculeLipid`, and `Pocket`. Each holds a `MoleculeCommon`, which carries the atoms, bonds, adjacency
list, and identifiers shared by all of them.

It also includes derived characterizations of a molecule, which are the inputs to property inference:

- `MolCharacterization` — rings, functional groups, rotatable bonds, and molecular descriptors
- `MolComponents` — a decomposition of a molecule into chemically meaningful components and their connections
- `Conformer` — a sampled conformational ensemble, with per-atom motion statistics
- `Pharmacophore` — H-bond donors and acceptors, hydrophobic sites, and ring centres in 3D

See [the docs](https://docs.rs/mol_defs) for details on the data structures and functions available.

## Features

- `render` — surface-mesh generation, and the mesh and electron-density fields used to draw molecules.
  Pulls in a GPU stack; leave it off for headless use such as ML inference.
