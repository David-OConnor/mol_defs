# DNA construction

`MoleculeNucleicAcid::from_seq` constructs an all-atom DNA strand or antiparallel
duplex from a sequence supplied in 5′→3′ order. DNA construction is implemented
in `dna.rs`.

Heavy-atom positions use the Arnott B-DNA fiber-diffraction model, with a rise of
3.38 Å and a twist of 36° per base pair. `b_dna_coordinates.rs` contains the
coordinate table and its source attribution. The coordinates, rise and twist
jointly determine backbone geometry; changing the helical parameters requires
refitting the backbone to preserve bond lengths and angles.

Amber OL24 templates supply atom names, connectivity, force-field types, charges
and hydrogen geometry. Hydrogens are placed in local frames defined by bonded
heavy atoms. Each strand has 5′ and 3′ hydroxyl termini. A single nucleotide uses
the corresponding neutral template (`DAN`, `DTN`, `DCN` or `DGN`).

Residues are stored in 5′→3′ order, with one residue per nucleotide. In a duplex,
the input strand is followed by its reverse complement. Atoms carry chain indices
0 and 1, and covalent bonds connect adjacent residues within each strand.
Single-stranded output has the same coordinates as the first strand of the
corresponding duplex. The helix axis is along Y.

The result is an idealized B-form starting structure. It does not model
sequence-dependent equilibrium geometry or the conformational ensemble of free
single-stranded DNA. Construction is deterministic and performs no energy
minimization or molecular dynamics.

Run `cargo test --lib nucleic_acid` to check geometry and topology.
