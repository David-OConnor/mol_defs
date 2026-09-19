//! Adding, removing, and detaching ligands (and other hetero residues: ions, cofactors etc.) on a
//! protein, keeping both the [`MoleculePeptide`] and the mmCIF it came from in sync.
//!
//! mmCIF files from the RCSB describe a bound ligand in many places besides its coordinates: its
//! entity and chemical component, the asym (instance) it occupies, its connections to protein atoms
//! (e.g. metal coordination), binding sites, and validation reports. We remove all of these along
//! with the ligand so the file stays consistent, and keep them on the detached ligand
//! ([`LigandCifOrigin`]) so re-attaching it can restore them.

use std::{
    collections::{HashMap, HashSet},
    io,
    io::ErrorKind,
};

use bio_files::{BondType, ResidueEnd, ResidueType};
use lin_alg::f64::Vec3;
use na_seq::{AtomTypeInRes, Element};

use crate::{
    mmcif_edit::{CifCategory, CifDoc, CifRow},
    molecules::{
        Atom, AtomRole, Bond, Chain, MolIdent, Residue, common::MoleculeCommon,
        peptide::MoleculePeptide, small::MoleculeSmall,
    },
    util::mol_center_size,
};

const WATER_COMPS: [&str; 3] = ["HOH", "WAT", "DOD"];

/// Categories describing the ligand instance itself. We restore these when re-attaching, with
/// their IDs updated if required. The rest of what we remove with a ligand describes its
/// relationship to the protein around it (connections, binding sites, validation), so we only
/// restore that when the ligand is returned unmoved to the protein it came from.
const INSTANCE_CATS: [&str; 5] = [
    "_atom_site",
    "_struct_asym",
    "_pdbx_nonpoly_scheme",
    "_pdbx_branch_scheme",
    "_pdbx_entity_instance_feature",
];

/// Distance, in Å, beyond which we consider a re-attached ligand's atom moved.
const MOVED_THRESH: f64 = 0.001;

/// Rows removed from, or copied out of, one mmCIF category.
#[derive(Clone, Debug)]
struct CifRows {
    category: String,
    tags: Vec<String>,
    is_loop: bool,
    header_raw: Option<String>,
    /// The nearest category before this one, for putting it back if it was removed entirely.
    anchor: Option<String>,
    /// The category's row count after removal. If it's unchanged when restoring, we put the rows
    /// back where they were.
    len_after: usize,
    /// With their original indices.
    rows: Vec<(usize, CifRow)>,
}

/// The mmCIF records of a ligand (or other hetero residue) detached from a protein. Re-attaching
/// the ligand restores what a plain atom list can't carry: its entity and chemical component
/// descriptions, atom names, B-factors, and alternate conformations. If it's returned unmoved to the
/// protein it came from, this also restores its connections, binding sites, and validation records.
#[derive(Clone, Debug, Default)]
pub struct LigandCifOrigin {
    /// Ident of the protein it was detached from.
    pub source_ident: String,
    pub label_asym_id: String,
    /// Usually `.`, for non-polymers.
    pub label_seq_id: String,
    pub entity_id: Option<String>,
    /// Chemical component IDs, e.g. "ATP". One, except for e.g. branched sugars.
    pub comp_ids: Vec<String>,
    /// (auth_asym_id, auth_seq_id, auth_comp_id) of each of its residues. E.g. ("A", "601", "DAD").
    pub auth_ids: Vec<(String, String, String)>,
    /// (assembly_id, oper_expression) of each `_pdbx_struct_assembly_gen` row that included it.
    assemblies: Vec<(String, String)>,
    /// Rows removed from the file, in the order removed. Includes its `_atom_site` rows.
    removed: Vec<CifRows>,
    /// Its entity and chemical component descriptions. Removed from the file along with it if
    /// nothing else used them; copied otherwise.
    descriptions: Vec<CifRows>,
}

/// A ligand's location in an mmCIF file.
#[derive(Clone, Debug, Default)]
struct HetInstance {
    label_asym_id: String,
    label_seq_id: String,
    entity_id: Option<String>,
    comp_ids: Vec<String>,
    /// (auth_asym_id, auth_seq_id, auth_comp_id)
    auth: Vec<(String, String, String)>,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg.into())
}

/// mmCIF's markers for unknown, and not-applicable values.
fn wild(v: &str) -> bool {
    v == "." || v == "?"
}

fn is_water(comp: &str) -> bool {
    WATER_COMPS.iter().any(|w| w.eq_ignore_ascii_case(comp))
}

fn is_instance_cat(name: &str) -> bool {
    INSTANCE_CATS.iter().any(|c| c.eq_ignore_ascii_case(name))
}

/// Categories describing entities, e.g. `_entity` and `_pdbx_entity_nonpoly`.
fn is_entity_cat(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("_entity") || n.starts_with("_pdbx_entity")
}

/// Categories describing chemical components, e.g. `_chem_comp` and `_chem_comp_bond`.
fn is_chem_comp_cat(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("_chem_comp") || n.starts_with("_pdbx_chem_comp")
}

/// The bio_files mmCIF parser keeps quotes on atom names that need them, e.g. `"O5'"`.
fn unquote(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn tag_col(tags: &[String], tag: &str) -> Option<usize> {
    tags.iter().position(|t| t.eq_ignore_ascii_case(tag))
}

/// All values in a column. Empty if the category or column is absent.
fn col_values<'a>(doc: &'a CifDoc, cat: &str, tag: &str) -> Vec<&'a str> {
    let Some(cat) = doc.category(cat) else {
        return Vec::new();
    };
    let Some(c) = cat.col(tag) else {
        return Vec::new();
    };

    cat.rows().iter().map(|r| r.get(c)).collect()
}

/// Build a row from a function of lowercase tag names; values it doesn't supply are `?`.
fn row_from(tags: &[String], f: impl Fn(&str) -> Option<String>) -> CifRow {
    CifRow::new(
        tags.iter()
            .map(|t| f(&t.to_ascii_lowercase()).unwrap_or_else(|| "?".to_owned()))
            .collect(),
    )
}

/// Map a row's values to another set of tags.
fn remap_row(row: &CifRow, from: &[String], to: &[String]) -> CifRow {
    CifRow::new(
        to.iter()
            .map(|t| match tag_col(from, t) {
                Some(c) => row.get(c).to_owned(),
                None => "?".to_owned(),
            })
            .collect(),
    )
}

fn set_tag(row: &mut CifRow, tags: &[String], tag: &str, value: &str) {
    if let Some(c) = tag_col(tags, tag) {
        row.set(c, value);
    }
}

fn cartn(row: &CifRow, cols: (usize, usize, usize)) -> Vec3 {
    let v = |c| row.get(c).parse().unwrap_or(0.);
    Vec3::new(v(cols.0), v(cols.1), v(cols.2))
}

/// Remove rows matching a predicate from a category.
fn take_rows(
    doc: &mut CifDoc,
    name: &str,
    mut pred: impl FnMut(&CifCategory, &CifRow) -> bool,
) -> Option<CifRows> {
    let anchor = doc.category_before(name);
    let cat = doc.category_mut(name)?;

    let indices: Vec<usize> = (0..cat.len())
        .filter(|&i| pred(cat, &cat.rows()[i]))
        .collect();
    if indices.is_empty() {
        return None;
    }
    let rows = cat.remove_rows(&indices);

    Some(CifRows {
        category: cat.name().to_owned(),
        tags: cat.tags().to_vec(),
        is_loop: cat.is_loop(),
        header_raw: cat.header_raw().cloned(),
        anchor,
        len_after: cat.len(),
        rows,
    })
}

/// Copy rows matching a predicate from a category, without removing them.
fn copy_rows(
    doc: &CifDoc,
    name: &str,
    mut pred: impl FnMut(&CifCategory, &CifRow) -> bool,
) -> Option<CifRows> {
    let cat = doc.category(name)?;

    let rows: Vec<(usize, CifRow)> = cat
        .rows()
        .iter()
        .enumerate()
        .filter(|(_, r)| pred(cat, r))
        .map(|(i, r)| (i, r.clone()))
        .collect();
    if rows.is_empty() {
        return None;
    }

    Some(CifRows {
        category: cat.name().to_owned(),
        tags: cat.tags().to_vec(),
        is_loop: cat.is_loop(),
        header_raw: cat.header_raw().cloned(),
        anchor: doc.category_before(name),
        len_after: cat.len(),
        rows,
    })
}

/// Put rows back into their category, adapting them to its tags if they differ. If the category
/// is gone, this re-creates it where it was when `create` is set, and skips the rows otherwise.
/// `edit` runs on each row (in the category's tag order) before insertion. `fallback` gives the
/// insertion index if the category has changed since, so the original indices no longer apply.
fn restore_rows(
    doc: &mut CifDoc,
    rec: &CifRows,
    create: bool,
    mut edit: impl FnMut(&[String], &mut CifRow),
    fallback: impl Fn(&CifCategory) -> usize,
) {
    if doc.category(&rec.category).is_none() {
        if !create {
            return;
        }
        let cat = CifCategory::from_parts(
            &rec.category,
            rec.tags.clone(),
            rec.is_loop,
            rec.header_raw.clone(),
            Vec::new(),
        );
        doc.insert_category(rec.anchor.as_deref(), cat);
    }
    let Some(cat) = doc.category_mut(&rec.category) else {
        return;
    };

    let in_place = cat.len() == rec.len_after;
    let will_be_loop = cat.is_loop() || rec.is_loop || cat.len() + rec.rows.len() > 1;
    if will_be_loop {
        cat.make_loop();
    }

    let tags = cat.tags().to_vec();
    let same_layout = rec.is_loop == will_be_loop
        && tags.len() == rec.tags.len()
        && tags
            .iter()
            .zip(&rec.tags)
            .all(|(a, b)| a.eq_ignore_ascii_case(b));

    for (i, row) in &rec.rows {
        let mut row = if same_layout {
            row.clone()
        } else {
            remap_row(row, &rec.tags, &tags)
        };
        edit(&tags, &mut row);

        let i = if in_place { *i } else { fallback(cat) };
        cat.insert_row(i, row);
    }
}

fn append(cat: &CifCategory) -> usize {
    cat.len()
}

/// The index after the last row that isn't water. RCSB files list waters last.
fn before_water(cat: &CifCategory, comp_tag: &str) -> usize {
    let c = cat.col(comp_tag);
    cat.rows()
        .iter()
        .rposition(|r| c.is_none_or(|c| !is_water(r.get(c))))
        .map(|i| i + 1)
        .unwrap_or(0)
}

/// The index at which to insert a row keyed `key` into a category sorted by the `tag` column.
fn sorted_index(cat: &CifCategory, tag: &str, key: &str) -> usize {
    let Some(c) = cat.col(tag) else {
        return cat.len();
    };
    let key = key.to_ascii_uppercase();
    cat.rows()
        .iter()
        .position(|r| r.get(c).to_ascii_uppercase() > key)
        .unwrap_or(cat.len())
}

/// Columns through which a category's rows can refer to a residue instance.
#[derive(Default)]
struct RefCols {
    /// (asym, seq, comp) columns, in label (mmCIF) numbering.
    label: Vec<(usize, Option<usize>, Option<usize>)>,
    /// (asym, seq, comp) columns, in author (PDB) numbering.
    auth: Vec<(usize, usize, usize)>,
}

impl RefCols {
    /// This works from naming conventions, so it covers categories we don't know about, e.g.
    /// `_struct_conn.ptnr2_label_asym_id`, `_struct_site.pdbx_auth_seq_id`, and
    /// `_pdbx_validate_close_contact.auth_comp_id_1`.
    fn new(cat: &CifCategory) -> Self {
        let mut result = Self::default();

        for (i, tag) in cat.tags().iter().enumerate() {
            let t = tag.to_ascii_lowercase();

            if t.contains("label_asym_id") {
                result.label.push((
                    i,
                    cat.col(&t.replace("label_asym_id", "label_seq_id")),
                    cat.col(&t.replace("label_asym_id", "label_comp_id")),
                ));
            } else if t.contains("auth_asym_id") {
                let seq = cat
                    .col(&t.replace("auth_asym_id", "auth_seq_id"))
                    .or_else(|| cat.col(&t.replace("auth_asym_id", "auth_seq_num")));
                let comp = cat
                    .col(&t.replace("auth_asym_id", "auth_comp_id"))
                    .or_else(|| cat.col(&t.replace("auth_asym_id", "auth_mon_id")));

                // Author chains hold the polymer too, so we need the residue number and type.
                if let (Some(s), Some(c)) = (seq, comp) {
                    result.auth.push((i, s, c));
                }
            } else if t == "asym_id" {
                result
                    .label
                    .push((i, None, cat.col("mon_id").or_else(|| cat.col("comp_id"))));
            }
        }

        result
    }

    fn is_empty(&self) -> bool {
        self.label.is_empty() && self.auth.is_empty()
    }

    fn refers_to(&self, row: &CifRow, inst: &HetInstance) -> bool {
        for &(c_asym, c_seq, c_comp) in &self.label {
            if row.get(c_asym) != inst.label_asym_id {
                continue;
            }

            // `None` if the column is absent, or the value unknown.
            let comp_match = c_comp.map(|c| row.get(c)).filter(|v| !wild(v)).map(|v| {
                inst.comp_ids
                    .iter()
                    .any(|comp| comp.eq_ignore_ascii_case(v))
            });
            if comp_match == Some(false) {
                continue;
            }

            let seq_ok = match c_seq {
                None => true,
                Some(c) => {
                    let v = row.get(c);
                    v == inst.label_seq_id
                        || (wild(v) && (wild(&inst.label_seq_id) || comp_match == Some(true)))
                }
            };
            if seq_ok {
                return true;
            }
        }

        for &(c_asym, c_seq, c_comp) in &self.auth {
            let (asym, seq, comp) = (row.get(c_asym), row.get(c_seq), row.get(c_comp));
            if inst
                .auth
                .iter()
                .any(|(a, s, c)| a == asym && s == seq && c.eq_ignore_ascii_case(comp))
            {
                return true;
            }
        }

        false
    }
}

/// Find where a residue of the peptide is in its mmCIF: by atom serial number, which the parser
/// takes from `_atom_site.id`, and failing that, by chain and residue name.
fn find_instance(
    doc: &CifDoc,
    atom_sns: &HashSet<u32>,
    chain_id: Option<&str>,
    res_name: &str,
) -> io::Result<HetInstance> {
    let site = doc
        .category("_atom_site")
        .filter(|c| !c.is_empty())
        .ok_or_else(|| invalid("The protein's mmCIF has no atoms (_atom_site)"))?;
    let col = |tag: &str| {
        site.col(tag)
            .ok_or_else(|| invalid(format!("The protein's mmCIF _atom_site has no {tag}")))
    };
    let (c_asym, c_seq, c_comp) = (
        col("label_asym_id")?,
        col("label_seq_id")?,
        col("label_comp_id")?,
    );
    let c_entity = site.col("label_entity_id");
    let (c_auth_asym, c_auth_seq, c_auth_comp) = (
        site.col("auth_asym_id"),
        site.col("auth_seq_id"),
        site.col("auth_comp_id"),
    );

    let mut votes: HashMap<(&str, &str), usize> = HashMap::new();
    if let Some(c_id) = site.col("id") {
        for row in site.rows() {
            if row
                .get(c_id)
                .parse::<u32>()
                .is_ok_and(|sn| atom_sns.contains(&sn))
            {
                *votes.entry((row.get(c_asym), row.get(c_seq))).or_default() += 1;
            }
        }
    }

    let key = votes
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k)
        .or_else(|| {
            let chain = chain_id?;
            site.rows()
                .iter()
                .find(|r| r.get(c_asym) == chain && r.get(c_comp).eq_ignore_ascii_case(res_name))
                .map(|r| (r.get(c_asym), r.get(c_seq)))
        });
    let Some((asym, seq)) = key else {
        return Err(invalid(format!(
            "Unable to find {res_name}'s atoms in the protein's mmCIF"
        )));
    };

    let mut result = HetInstance {
        label_asym_id: asym.to_owned(),
        label_seq_id: seq.to_owned(),
        ..Default::default()
    };

    for row in site.rows() {
        if row.get(c_asym) != asym || row.get(c_seq) != seq {
            continue;
        }

        let comp = row.get(c_comp);
        if !result.comp_ids.iter().any(|c| c == comp) {
            result.comp_ids.push(comp.to_owned());
        }

        if result.entity_id.is_none()
            && let Some(c) = c_entity
            && !wild(row.get(c))
        {
            result.entity_id = Some(row.get(c).to_owned());
        }

        if let (Some(a), Some(s)) = (c_auth_asym, c_auth_seq) {
            let auth_comp = c_auth_comp.map(|c| row.get(c)).unwrap_or(comp);
            let auth = (
                row.get(a).to_owned(),
                row.get(s).to_owned(),
                auth_comp.to_owned(),
            );
            if !result.auth.contains(&auth) {
                result.auth.push(auth);
            }
        }
    }

    if result.comp_ids.iter().all(|c| is_water(c)) {
        return Err(invalid("This is water, not a ligand"));
    }

    // A polymer residue is also described by the polymer's sequence; removing its atoms alone
    // would leave that inconsistent.
    let polymer = col_values(doc, "_pdbx_poly_seq_scheme", "asym_id").contains(&asym)
        || result
            .entity_id
            .as_ref()
            .is_some_and(|e| col_values(doc, "_entity_poly", "entity_id").contains(&e.as_str()));
    if polymer && !wild(seq) {
        return Err(invalid(format!(
            "{} is part of a polymer chain, so can't be removed as a ligand",
            result.comp_ids.join(", ")
        )));
    }

    Ok(result)
}

/// Remove a residue instance, and everything referring to it, from an mmCIF document.
fn remove_instance(doc: &mut CifDoc, inst: &HetInstance, source_ident: &str) -> LigandCifOrigin {
    let mut removed = Vec::new();

    let mut names: Vec<String> = Vec::new();
    for cat in doc.categories() {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(cat.name())) {
            names.push(cat.name().to_owned());
        }
    }

    // Its atoms, and anything referring to it: connections, binding sites, validation etc.
    for name in &names {
        // We handle this below, as other residues may share its asym.
        if name.eq_ignore_ascii_case("_struct_asym") {
            continue;
        }
        let Some(cat) = doc.category(name) else {
            continue;
        };
        let refs = RefCols::new(cat);
        if refs.is_empty() {
            continue;
        }

        if let Some(r) = take_rows(doc, name, |_, row| refs.refers_to(row, inst)) {
            removed.push(r);
        }
    }

    // Binding sites for this ligand go with it, along with their residue lists.
    let site_ids = col_values_of(&removed, "_struct_site", "id");
    if !site_ids.is_empty()
        && let Some(r) = take_rows(doc, "_struct_site_gen", |cat, row| {
            cat.col("site_id")
                .is_some_and(|c| site_ids.contains(row.get(c)))
        })
    {
        removed.push(r);
    }

    let asym = inst.label_asym_id.as_str();
    let asym_gone = !col_values(doc, "_atom_site", "label_asym_id").contains(&asym);

    let mut assemblies = Vec::new();
    if asym_gone {
        if let Some(r) = take_rows(doc, "_struct_asym", |cat, row| {
            cat.col("id").is_some_and(|c| row.get(c) == asym)
        }) {
            removed.push(r);
        }
        assemblies = remove_from_assemblies(doc, asym);
    }

    // Its entity and chemical component: removed if nothing else is of them; copied either way.
    let mut descriptions = Vec::new();
    if let Some(entity) = &inst.entity_id {
        let in_use = col_values(doc, "_atom_site", "label_entity_id").contains(&entity.as_str())
            || col_values(doc, "_struct_asym", "entity_id").contains(&entity.as_str());

        descriptions.extend(description_rows(doc, !in_use, is_entity_cat, |cat, row| {
            let key = if cat.name().eq_ignore_ascii_case("_entity") {
                "id"
            } else {
                "entity_id"
            };
            cat.col(key).is_some_and(|c| row.get(c) == entity.as_str())
        }));

        if in_use && asym_gone {
            adjust_entity_count(doc, entity, -1);
        }
    }

    for comp in &inst.comp_ids {
        let used = |cat: &str, tag: &str| {
            col_values(doc, cat, tag)
                .iter()
                .any(|v| v.eq_ignore_ascii_case(comp))
        };
        let in_use = used("_atom_site", "label_comp_id")
            || used("_pdbx_poly_seq_scheme", "mon_id")
            || used("_entity_poly_seq", "mon_id");

        descriptions.extend(description_rows(
            doc,
            !in_use,
            is_chem_comp_cat,
            |cat, row| {
                let key = if cat.name().eq_ignore_ascii_case("_chem_comp") {
                    "id"
                } else {
                    "comp_id"
                };
                cat.col(key)
                    .is_some_and(|c| row.get(c).eq_ignore_ascii_case(comp))
            },
        ));
    }

    // Connection types nothing uses anymore.
    if doc.category("_struct_conn").is_some() {
        let used: HashSet<String> = col_values(doc, "_struct_conn", "conn_type_id")
            .iter()
            .map(|v| v.to_ascii_lowercase())
            .collect();

        if let Some(r) = take_rows(doc, "_struct_conn_type", |cat, row| {
            cat.col("id")
                .is_some_and(|c| !used.contains(&row.get(c).to_ascii_lowercase()))
        }) {
            removed.push(r);
        }
    }

    recount_sites(doc, &col_values_of(&removed, "_struct_site_gen", "site_id"));
    prune_atom_types(doc);

    LigandCifOrigin {
        source_ident: source_ident.to_owned(),
        label_asym_id: inst.label_asym_id.clone(),
        label_seq_id: inst.label_seq_id.clone(),
        entity_id: inst.entity_id.clone(),
        comp_ids: inst.comp_ids.clone(),
        auth_ids: inst.auth.clone(),
        assemblies,
        removed,
        descriptions,
    }
}

/// Values of a column, across removed rows of a category.
fn col_values_of(recs: &[CifRows], cat: &str, tag: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    for rec in recs.iter().filter(|r| r.category.eq_ignore_ascii_case(cat)) {
        if let Some(c) = tag_col(&rec.tags, tag) {
            result.extend(rec.rows.iter().map(|(_, row)| row.get(c).to_owned()));
        }
    }
    result
}

/// Remove (or copy) rows in description categories, e.g. those of an entity.
fn description_rows(
    doc: &mut CifDoc,
    remove: bool,
    is_cat: fn(&str) -> bool,
    pred: impl Fn(&CifCategory, &CifRow) -> bool,
) -> Vec<CifRows> {
    // Categories that refer to instances (e.g. `_pdbx_entity_instance_feature`) aren't
    // descriptions; we handle those along with the instance.
    let names: Vec<String> = doc
        .categories()
        .filter(|c| is_cat(c.name()) && RefCols::new(c).is_empty())
        .map(|c| c.name().to_owned())
        .collect();

    names
        .iter()
        .filter_map(|name| {
            if remove {
                take_rows(doc, name, &pred)
            } else {
                copy_rows(doc, name, &pred)
            }
        })
        .collect()
}

/// Remove an asym from the biological assemblies listing it; returns those.
fn remove_from_assemblies(doc: &mut CifDoc, asym: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();

    let Some(cat) = doc.category_mut("_pdbx_struct_assembly_gen") else {
        return result;
    };
    let Some(c_list) = cat.col("asym_id_list") else {
        return result;
    };

    for i in 0..cat.len() {
        let list = cat.rows()[i].get(c_list).to_owned();
        let parts: Vec<&str> = list.split(',').map(str::trim).collect();
        if !parts.contains(&asym) {
            continue;
        }

        result.push((
            cat.get(i, "assembly_id").unwrap_or_default().to_owned(),
            cat.get(i, "oper_expression").unwrap_or_default().to_owned(),
        ));

        let remaining: Vec<&str> = parts.into_iter().filter(|p| *p != asym).collect();
        cat.set_col(i, c_list, &remaining.join(","));
    }

    result
}

/// Sort key for asym IDs, as the PDB assigns them: A-Z, then AA, BA, CA ... (first letter fastest).
fn asym_key(id: &str) -> (usize, String) {
    (id.len(), id.chars().rev().collect())
}

/// Add an asym to biological assemblies: the ones given, or failing that, those holding the
/// polymer chain it's nearest.
fn add_to_assemblies(
    doc: &mut CifDoc,
    asym: &str,
    targets: &[(String, String)],
    nearest_asym: Option<&str>,
) {
    let Some(cat) = doc.category_mut("_pdbx_struct_assembly_gen") else {
        return;
    };
    let Some(c_list) = cat.col("asym_id_list") else {
        return;
    };

    let lists: Vec<Vec<String>> = cat
        .rows()
        .iter()
        .map(|r| {
            r.get(c_list)
                .split(',')
                .map(|a| a.trim().to_owned())
                .collect()
        })
        .collect();

    let mut rows: Vec<usize> = (0..cat.len())
        .filter(|&i| {
            targets.iter().any(|(id, oper)| {
                cat.get(i, "assembly_id").unwrap_or_default() == id
                    && cat.get(i, "oper_expression").unwrap_or_default() == oper
            })
        })
        .collect();
    if rows.is_empty()
        && let Some(n) = nearest_asym
    {
        rows = (0..cat.len())
            .filter(|&i| lists[i].iter().any(|a| a == n))
            .collect();
    }
    if rows.is_empty() && cat.len() == 1 {
        rows.push(0);
    }

    for i in rows {
        let mut list = lists[i].clone();
        list.retain(|a| !a.is_empty());
        if list.iter().any(|a| a == asym) {
            continue;
        }

        let sorted = list.windows(2).all(|w| asym_key(&w[0]) <= asym_key(&w[1]));
        let pos = if sorted {
            list.iter()
                .position(|a| asym_key(a) > asym_key(asym))
                .unwrap_or(list.len())
        } else {
            list.len()
        };
        list.insert(pos, asym.to_owned());

        cat.set_col(i, c_list, &list.join(","));
    }
}

fn adjust_entity_count(doc: &mut CifDoc, entity: &str, delta: i64) {
    let Some(cat) = doc.category_mut("_entity") else {
        return;
    };
    let (Some(c_id), Some(c_n)) = (cat.col("id"), cat.col("pdbx_number_of_molecules")) else {
        return;
    };

    for i in 0..cat.len() {
        let row = &cat.rows()[i];
        if row.get(c_id) == entity
            && let Ok(n) = row.get(c_n).parse::<i64>()
        {
            cat.set_col(i, c_n, &(n + delta).max(0).to_string());
        }
    }
}

/// Update the residue counts of binding sites whose residue lists we've changed.
fn recount_sites(doc: &mut CifDoc, site_ids: &HashSet<String>) {
    if site_ids.is_empty() {
        return;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for id in col_values(doc, "_struct_site_gen", "site_id") {
        *counts.entry(id.to_owned()).or_default() += 1;
    }

    let Some(cat) = doc.category_mut("_struct_site") else {
        return;
    };
    let (Some(c_id), Some(c_n)) = (cat.col("id"), cat.col("pdbx_num_residues")) else {
        return;
    };

    for i in 0..cat.len() {
        let row = &cat.rows()[i];
        let id = row.get(c_id);
        if site_ids.contains(id) && row.get(c_n).parse::<usize>().is_ok() {
            let n = counts.get(id).copied().unwrap_or(0);
            cat.set_col(i, c_n, &n.to_string());
        }
    }
}

/// Remove elements from `_atom_type` that no atoms are anymore.
fn prune_atom_types(doc: &mut CifDoc) {
    let present: HashSet<String> = col_values(doc, "_atom_site", "type_symbol")
        .iter()
        .map(|s| s.to_ascii_uppercase())
        .collect();
    if present.is_empty() {
        return;
    }

    take_rows(doc, "_atom_type", |cat, row| {
        cat.col("symbol")
            .is_some_and(|c| !present.contains(&row.get(c).to_ascii_uppercase()))
    });
}

fn add_atom_types(doc: &mut CifDoc, symbols: &[String]) {
    let Some(cat) = doc.category_mut("_atom_type") else {
        return;
    };
    let Some(c) = cat.col("symbol") else {
        return;
    };

    for sym in symbols {
        if cat
            .rows()
            .iter()
            .any(|r| r.get(c).eq_ignore_ascii_case(sym))
        {
            continue;
        }
        let i = sorted_index(cat, "symbol", sym);
        let tags = cat.tags().to_vec();
        cat.insert_row(i, row_from(&tags, |t| (t == "symbol").then(|| sym.clone())));
    }
}

/// The entity of a chemical component already in the file, if any.
fn entity_for_comp(doc: &CifDoc, comp: &str) -> Option<String> {
    if let Some(cat) = doc.category("_pdbx_entity_nonpoly")
        && let (Some(c_entity), Some(c_comp)) = (cat.col("entity_id"), cat.col("comp_id"))
        && let Some(row) = cat
            .rows()
            .iter()
            .find(|r| r.get(c_comp).eq_ignore_ascii_case(comp))
    {
        return Some(row.get(c_entity).to_owned());
    }

    let site = doc.category("_atom_site")?;
    let (c_entity, c_comp) = (site.col("label_entity_id")?, site.col("label_comp_id")?);
    site.rows()
        .iter()
        .find(|r| r.get(c_comp).eq_ignore_ascii_case(comp) && !wild(r.get(c_entity)))
        .map(|r| r.get(c_entity).to_owned())
}

fn entity_ids(doc: &CifDoc) -> HashSet<String> {
    let mut result: HashSet<String> = HashSet::new();
    for (cat, tag) in [
        ("_entity", "id"),
        ("_atom_site", "label_entity_id"),
        ("_struct_asym", "entity_id"),
    ] {
        result.extend(col_values(doc, cat, tag).iter().map(|v| v.to_string()));
    }
    result
}

/// Asym IDs in the order the PDB assigns them: A-Z, then AA, BA, CA ... ZA, AB, BB ...
fn next_asym_id(used: &HashSet<String>) -> String {
    let letters: Vec<char> = ('A'..='Z').collect();

    for &a in &letters {
        let id = a.to_string();
        if !used.contains(&id) {
            return id;
        }
    }
    for &b in &letters {
        for &a in &letters {
            let id = format!("{a}{b}");
            if !used.contains(&id) {
                return id;
            }
        }
    }
    for &c in &letters {
        for &b in &letters {
            for &a in &letters {
                let id = format!("{a}{b}{c}");
                if !used.contains(&id) {
                    return id;
                }
            }
        }
    }

    "ZZZZ".to_owned()
}

/// Whether the file includes hydrogen atoms. X-ray structures generally don't; if this one
/// doesn't, we leave them off ligands we add, for consistency.
fn doc_has_hydrogen(doc: &CifDoc) -> bool {
    col_values(doc, "_atom_site", "type_symbol")
        .iter()
        .any(|s| s.eq_ignore_ascii_case("H") || s.eq_ignore_ascii_case("D"))
}

/// Values of each column shared by all rows, e.g. `pdbx_PDB_model_num`. Used for columns we don't
/// otherwise know how to fill.
fn constant_values(cat: &CifCategory) -> Vec<Option<String>> {
    (0..cat.tags().len())
        .map(|c| {
            let first = cat.rows().first()?.get(c);
            cat.rows()
                .iter()
                .all(|r| r.get(c) == first)
                .then(|| first.to_owned())
        })
        .collect()
}

/// The polymer chain nearest a point, as (label_asym_id, auth_asym_id).
fn nearest_polymer_chain(site: &CifCategory, posit: Vec3) -> Option<(String, String)> {
    let cols = (
        site.col("Cartn_x")?,
        site.col("Cartn_y")?,
        site.col("Cartn_z")?,
    );
    let c_asym = site.col("label_asym_id")?;
    let c_auth = site.col("auth_asym_id").unwrap_or(c_asym);
    let c_group = site.col("group_PDB");
    let c_comp = site.col("label_comp_id")?;

    let mut result = None;
    let mut best = f64::MAX;
    for row in site.rows() {
        let polymer = match c_group {
            Some(g) => row.get(g) == "ATOM",
            None => matches!(
                ResidueType::from_str(row.get(c_comp)),
                ResidueType::AminoAcid(_)
            ),
        };
        if !polymer {
            continue;
        }

        let dist = (cartn(row, cols) - posit).magnitude_squared();
        if dist < best {
            best = dist;
            result = Some((row.get(c_asym).to_owned(), row.get(c_auth).to_owned()));
        }
    }

    result
}

/// IDs of a ligand being restored: what they were when detached, and what they are now.
#[derive(Default)]
struct IdMap {
    old_asym: String,
    new_asym: String,
    old_entity: String,
    new_entity: String,
    old_auth_asym: String,
    new_auth_asym: String,
    /// Old auth_seq_id to new.
    seqs: HashMap<String, String>,
    /// Old comp ID to new, if renamed.
    comps: HashMap<String, String>,
}

impl IdMap {
    fn unchanged(&self) -> bool {
        self.old_asym == self.new_asym
            && self.old_entity == self.new_entity
            && self.old_auth_asym == self.new_auth_asym
            && self.seqs.iter().all(|(a, b)| a == b)
            && self.comps.is_empty()
    }
}

/// What a category's `id` column holds, for [`rewrite_ids`].
#[derive(Clone, Copy, PartialEq)]
enum IdKind {
    Other,
    Asym,
    Entity,
    Comp,
}

impl IdKind {
    fn of(category: &str) -> Self {
        match category.to_ascii_lowercase().as_str() {
            "_struct_asym" => Self::Asym,
            "_entity" => Self::Entity,
            "_chem_comp" => Self::Comp,
            _ => Self::Other,
        }
    }
}

/// Update a restored row's asym, entity, author, and component IDs, identified by tag names.
fn rewrite_ids(tags: &[String], row: &mut CifRow, ids: &IdMap, id_kind: IdKind) {
    let swap =
        |v: &str, old: &str, new: &str| (!old.is_empty() && v == old).then(|| new.to_owned());

    for (i, tag) in tags.iter().enumerate() {
        let t = tag.to_ascii_lowercase();
        let v = row.get(i).to_owned();

        let new = if t == "id" {
            match id_kind {
                IdKind::Asym => swap(&v, &ids.old_asym, &ids.new_asym),
                IdKind::Entity => swap(&v, &ids.old_entity, &ids.new_entity),
                IdKind::Comp => ids.comps.get(&v).cloned(),
                IdKind::Other => None,
            }
        } else if t.contains("entity_id") {
            swap(&v, &ids.old_entity, &ids.new_entity)
        } else if t.contains("comp_id") || t.contains("mon_id") {
            ids.comps.get(&v).cloned()
        } else if t.contains("auth_asym_id") || t.contains("strand_id") || t == "pdb_asym_id" {
            swap(&v, &ids.old_auth_asym, &ids.new_auth_asym)
        } else if t.contains("asym_id") {
            swap(&v, &ids.old_asym, &ids.new_asym)
        } else if t.contains("auth_seq") || t == "pdb_seq_num" {
            ids.seqs.get(&v).cloned()
        } else {
            None
        };

        if let Some(new) = new {
            row.set(i, &new);
        }
    }
}

/// Hill-notation formula, as in `_chem_comp.formula`, e.g. "C10 H16 N5 O11 P3"; and weight.
fn formula_weight(atoms: &[Atom]) -> (String, f32) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut weight = 0.;
    for atom in atoms {
        *counts.entry(atom.element.to_letter()).or_default() += 1;
        weight += atom.element.atomic_weight();
    }

    let mut symbols: Vec<&String> = counts.keys().collect();
    symbols.sort_by_key(|s| {
        let rank = match (s.as_str(), counts.contains_key("C")) {
            ("C", true) => 0,
            ("H", true) => 1,
            _ => 2,
        };
        (rank, s.to_string())
    });

    let formula = symbols
        .iter()
        .map(|s| match counts[*s] {
            1 => s.to_string(),
            n => format!("{s}{n}"),
        })
        .collect::<Vec<_>>()
        .join(" ");

    (formula, weight)
}

/// `_chem_comp_bond.value_order`, and whether it's aromatic.
fn bond_order(bond_type: BondType) -> (&'static str, bool) {
    match bond_type {
        BondType::Double => ("DOUB", false),
        BondType::Triple => ("TRIP", false),
        BondType::Aromatic => ("AROM", true),
        BondType::Quadruple => ("QUAD", false),
        BondType::Delocalized => ("DELO", false),
        _ => ("SING", false),
    }
}

/// Unique atom names for a ligand, e.g. "C1", "O2'": from the atoms' names where present and
/// unique, and generated from their elements otherwise.
fn ligand_atom_names(mol: &MoleculeCommon) -> Vec<String> {
    let mut used = HashSet::new();

    let names: Vec<Option<String>> = mol
        .atoms
        .iter()
        .map(|a| {
            let name = match &a.type_in_res {
                Some(AtomTypeInRes::Hetero(n)) => unquote(n.trim()).to_owned(),
                Some(t) => t.to_string(),
                None => a.type_in_res_general.clone().unwrap_or_default(),
            };
            let name = name.trim().to_owned();

            (!name.is_empty() && !name.contains(char::is_whitespace) && used.insert(name.clone()))
                .then_some(name)
        })
        .collect();

    let mut counts: HashMap<String, usize> = HashMap::new();
    names
        .into_iter()
        .zip(&mol.atoms)
        .map(|(name, atom)| {
            name.unwrap_or_else(|| {
                let sym = atom.element.to_letter().to_uppercase();
                loop {
                    let n = counts.entry(sym.clone()).or_default();
                    *n += 1;
                    let candidate = format!("{sym}{n}");
                    if used.insert(candidate.clone()) {
                        break candidate;
                    }
                }
            })
        })
        .collect()
}

/// A name for a ligand's entity and chemical component records.
fn ligand_descrip(lig: &MoleculeSmall) -> String {
    for ident in &lig.idents {
        if let MolIdent::PubchemTitle(name) | MolIdent::IupacName(name) = ident
            && !name.is_empty()
        {
            return name.clone();
        }
    }
    lig.common.ident.clone()
}

/// Validate a chemical component ID, e.g. "ATP": 1-5 letters and digits, which we uppercase.
fn normalize_comp_id(comp: &str) -> io::Result<String> {
    let comp = comp.trim().to_ascii_uppercase();
    if comp.is_empty() || comp.len() > 5 || !comp.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(invalid(format!(
            "Invalid residue name {comp:?}: use 1-5 letters and digits, e.g. \"LIG\""
        )));
    }
    if is_water(&comp) {
        return Err(invalid("A ligand can't use water's residue name"));
    }
    Ok(comp)
}

/// Maps world positions back to a molecule's own (file) coordinates, for one that may have been
/// moved or rotated as a rigid body after loading.
struct RigidFrame {
    moved: bool,
    origin_local: Vec3,
    origin_world: Vec3,
    axes_local: [Vec3; 3],
    axes_world: [Vec3; 3],
}

impl RigidFrame {
    fn new(mol: &MoleculeCommon) -> Self {
        let n = mol.atoms.len().min(mol.atom_posits.len());
        let local = |i: usize| mol.atoms[i].posit;
        let world = |i: usize| mol.atom_posits[i];

        let unit = [
            Vec3::new(1., 0., 0.),
            Vec3::new(0., 1., 0.),
            Vec3::new(0., 0., 1.),
        ];
        let mut result = Self {
            moved: false,
            origin_local: Vec3::new_zero(),
            origin_world: Vec3::new_zero(),
            axes_local: unit,
            axes_world: unit,
        };

        if n == 0 || (0..n).all(|i| (world(i) - local(i)).magnitude_squared() < 1e-12) {
            return result;
        }
        result.moved = true;
        result.origin_local = local(0);
        result.origin_world = world(0);

        // Build matching orthonormal frames from three well-separated atoms.
        let far =
            |f: &dyn Fn(usize) -> f64| (0..n).max_by(|&a, &b| f(a).total_cmp(&f(b))).unwrap_or(0);
        let i1 = far(&|i| (local(i) - local(0)).magnitude_squared());
        let d1 = local(i1) - local(0);
        if d1.magnitude() < 1e-6 {
            // A single point: translation only.
            return result;
        }
        let e1 = d1.to_normalized();
        let i2 = far(&|i| (local(i) - local(0)).cross(e1).magnitude_squared());
        if (local(i2) - local(0)).cross(e1).magnitude() < 1e-6 {
            // Colinear atoms; we can't recover a rotation about their axis.
            return result;
        }

        let axes = |p: &dyn Fn(usize) -> Vec3| {
            let a = (p(i1) - p(0)).to_normalized();
            let v = p(i2) - p(0);
            let b = (v - a * v.dot(a)).to_normalized();
            [a, b, a.cross(b)]
        };
        result.axes_local = axes(&local);
        result.axes_world = axes(&world);

        result
    }

    fn to_local(&self, p: Vec3) -> Vec3 {
        if !self.moved {
            return p;
        }
        let d = p - self.origin_world;
        let mut result = self.origin_local;
        for k in 0..3 {
            result += self.axes_local[k] * d.dot(self.axes_world[k]);
        }
        result
    }
}

/// Update metadata values the mmCIF parser takes from key-value items, for categories we've changed.
fn sync_metadata(md: &mut HashMap<String, String>, doc: &CifDoc) {
    for cat in doc.categories().filter(|c| c.is_modified()) {
        let prefix = format!("{}.", cat.name());
        md.retain(|k, _| !k.starts_with(&prefix));

        if !cat.is_loop() && cat.len() == 1 {
            for (tag, v) in cat.tags().iter().zip(cat.rows()[0].values()) {
                let v = if v.contains('\n') { "" } else { v };
                md.insert(format!("{prefix}{tag}"), v.to_owned());
            }
        }
    }
}

struct AttachInput<'a> {
    lig: &'a MoleculeSmall,
    names: &'a [String],
    /// In the protein's (file) coordinates.
    posits: &'a [Vec3],
    comp_id: &'a str,
    include_h: bool,
    target_ident: &'a str,
    /// Serial numbers of the peptide's atoms; the parser uses `_atom_site.id` for these.
    used_sns: &'a HashSet<u32>,
}

struct AttachOutput {
    /// (ligand atom index, serial number) of each atom added.
    atoms: Vec<(usize, u32)>,
    label_asym_id: String,
    label_seq: u32,
    comp_id: String,
}

/// Add a ligand to an mmCIF document, as a new non-polymer instance. If the ligand was detached
/// from a protein, this restores its records.
fn attach_to_doc(doc: &mut CifDoc, inp: &AttachInput) -> io::Result<AttachOutput> {
    let origin = inp.lig.cif_origin.as_ref();
    let atoms = &inp.lig.common.atoms;

    let site = doc
        .category("_atom_site")
        .filter(|c| !c.is_empty())
        .ok_or_else(|| invalid("The protein's mmCIF has no atoms (_atom_site)"))?;
    for tag in [
        "id",
        "type_symbol",
        "label_atom_id",
        "label_comp_id",
        "label_asym_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ] {
        if site.col(tag).is_none() {
            return Err(invalid(format!(
                "The protein's mmCIF _atom_site has no {tag}"
            )));
        }
    }
    let site_tags = site.tags().to_vec();
    let constants = constant_values(site);

    // What's already in the file.
    let mut used_asyms: HashSet<String> = HashSet::new();
    for (cat, tag) in [("_atom_site", "label_asym_id"), ("_struct_asym", "id")] {
        used_asyms.extend(col_values(doc, cat, tag).iter().map(|v| v.to_string()));
    }

    let mut used_ids: HashSet<u32> = inp.used_sns.clone();
    used_ids.extend(
        col_values(doc, "_atom_site", "id")
            .iter()
            .filter_map(|v| v.parse::<u32>().ok()),
    );

    let mut auth_used: HashSet<(String, String)> = HashSet::new();
    let mut max_auth_seq: HashMap<String, i64> = HashMap::new();
    if let (Some(c_a), Some(c_s)) = (site.col("auth_asym_id"), site.col("auth_seq_id")) {
        for row in site.rows() {
            let (a, s) = (row.get(c_a), row.get(c_s));
            auth_used.insert((a.to_owned(), s.to_owned()));

            let max = max_auth_seq.entry(a.to_owned()).or_insert(0);
            if let Ok(s) = s.parse::<i64>() {
                *max = (*max).max(s);
            }
        }
    }

    let centroid = inp.posits.iter().fold(Vec3::new_zero(), |acc, p| acc + *p)
        / inp.posits.len().max(1) as f64;
    let nearest = nearest_polymer_chain(site, centroid);

    let same_protein =
        origin.is_some_and(|o| o.source_ident.eq_ignore_ascii_case(inp.target_ident));

    // Choose IDs, keeping the ligand's original ones where they're free.
    let mut ids = IdMap::default();

    ids.new_asym = match origin {
        Some(o) if !o.label_asym_id.is_empty() && !used_asyms.contains(&o.label_asym_id) => {
            o.label_asym_id.clone()
        }
        _ => next_asym_id(&used_asyms),
    };

    // The requested component ID renames the ligand's own, if it has just one.
    let comp = match origin {
        Some(o) if o.comp_ids.len() == 1 => {
            if !o.comp_ids[0].eq_ignore_ascii_case(inp.comp_id) {
                ids.comps
                    .insert(o.comp_ids[0].clone(), inp.comp_id.to_owned());
            }
            inp.comp_id.to_owned()
        }
        Some(o) if !o.comp_ids.is_empty() => o.comp_ids[0].clone(),
        _ => inp.comp_id.to_owned(),
    };

    let existing_entity = entity_for_comp(doc, &comp);
    let new_entity = existing_entity.is_none();
    ids.new_entity = match existing_entity {
        Some(e) => e,
        None => {
            let used = entity_ids(doc);
            match origin.and_then(|o| o.entity_id.clone()) {
                Some(e) if !used.contains(&e) => e,
                _ => {
                    let max = used.iter().filter_map(|e| e.parse::<u32>().ok()).max();
                    (max.unwrap_or(0) + 1).to_string()
                }
            }
        }
    };

    ids.new_auth_asym = origin
        .and_then(|o| o.auth_ids.first())
        .map(|a| a.0.clone())
        .filter(|a| max_auth_seq.contains_key(a))
        .or_else(|| nearest.as_ref().map(|n| n.1.clone()))
        .unwrap_or_else(|| ids.new_asym.clone());

    let mut taken_seqs: HashSet<String> = auth_used
        .iter()
        .filter(|(a, _)| *a == ids.new_auth_asym)
        .map(|(_, s)| s.clone())
        .collect();
    let mut next_seq = max_auth_seq.get(&ids.new_auth_asym).copied().unwrap_or(0) + 1;
    let mut alloc_seq = |preferred: Option<&str>| -> String {
        if let Some(p) = preferred
            && taken_seqs.insert(p.to_owned())
        {
            return p.to_owned();
        }
        while !taken_seqs.insert(next_seq.to_string()) {
            next_seq += 1;
        }
        next_seq.to_string()
    };

    let auth_seq = match origin {
        Some(o) if !o.auth_ids.is_empty() => {
            ids.old_asym = o.label_asym_id.clone();
            ids.old_entity = o.entity_id.clone().unwrap_or_default();
            ids.old_auth_asym = o.auth_ids[0].0.clone();

            for (_, seq, _) in &o.auth_ids {
                if !ids.seqs.contains_key(seq) {
                    let new = alloc_seq(Some(seq));
                    ids.seqs.insert(seq.to_owned(), new);
                }
            }
            ids.seqs[&o.auth_ids[0].1].clone()
        }
        _ => alloc_seq(None),
    };

    // Atoms. Those the ligand had when detached keep their rows (and so their B-factors,
    // alternate conformations etc.), with positions updated. We match them by name.
    let origin_site = origin.and_then(|o| {
        o.removed
            .iter()
            .find(|r| r.category.eq_ignore_ascii_case("_atom_site"))
    });
    let (o_tags, o_rows): (&[String], &[(usize, CifRow)]) = match origin_site {
        Some(r) => (&r.tags, &r.rows),
        None => (&[], &[]),
    };
    let o_name = tag_col(o_tags, "label_atom_id");
    let o_cartn = match (
        tag_col(o_tags, "Cartn_x"),
        tag_col(o_tags, "Cartn_y"),
        tag_col(o_tags, "Cartn_z"),
    ) {
        (Some(x), Some(y), Some(z)) => Some((x, y, z)),
        _ => None,
    };
    let o_id = tag_col(o_tags, "id");

    let mut next_id = used_ids.iter().max().copied().unwrap_or(0) + 1;
    let mut alloc_id = |preferred: Option<u32>| -> u32 {
        if let Some(p) = preferred
            && used_ids.insert(p)
        {
            return p;
        }
        while !used_ids.insert(next_id) {
            next_id += 1;
        }
        next_id
    };

    let symbol = |a: &Atom| a.element.to_letter().to_uppercase();

    let mut new_rows: Vec<(Option<usize>, CifRow)> = Vec::new();
    let mut atoms_out = Vec::new();
    let mut matched = HashSet::new();
    let mut moved = false;

    for (i, atom) in atoms.iter().enumerate() {
        if !inp.include_h && atom.element == Element::Hydrogen {
            continue;
        }
        let name = &inp.names[i];
        let posit = inp.posits[i];

        let matches: Vec<usize> = match o_name {
            Some(c) => (0..o_rows.len())
                .filter(|k| !matched.contains(k) && o_rows[*k].1.get(c) == name.as_str())
                .collect(),
            None => Vec::new(),
        };

        if let (Some(&first), Some(cols)) = (matches.first(), o_cartn) {
            let delta = posit - cartn(&o_rows[first].1, cols);
            if delta.magnitude() > MOVED_THRESH {
                moved = true;
            }

            let mut sn = 0;
            for &k in &matches {
                matched.insert(k);
                let (orig_i, orig) = &o_rows[k];

                let mut row = if o_tags.len() == site_tags.len()
                    && o_tags
                        .iter()
                        .zip(&site_tags)
                        .all(|(a, b)| a.eq_ignore_ascii_case(b))
                {
                    orig.clone()
                } else {
                    remap_row(orig, o_tags, &site_tags)
                };

                // Shift alternate conformations along with the one we have.
                if delta.magnitude() > 1e-6 {
                    let p = cartn(orig, cols) + delta;
                    set_tag(&mut row, &site_tags, "Cartn_x", &format!("{:.3}", p.x));
                    set_tag(&mut row, &site_tags, "Cartn_y", &format!("{:.3}", p.y));
                    set_tag(&mut row, &site_tags, "Cartn_z", &format!("{:.3}", p.z));
                }
                rewrite_ids(&site_tags, &mut row, &ids, IdKind::Other);

                let id = alloc_id(o_id.and_then(|c| orig.get(c).parse().ok()));
                set_tag(&mut row, &site_tags, "id", &id.to_string());
                if sn == 0 {
                    sn = id;
                }

                new_rows.push((Some(*orig_i), row));
            }
            atoms_out.push((i, sn));
        } else {
            let id = alloc_id(None);
            let row = row_from(&site_tags, |t| {
                let col = site_tags.iter().position(|s| s.eq_ignore_ascii_case(t));
                let constant = col.and_then(|c| constants[c].clone());

                Some(match t {
                    "group_pdb" => "HETATM".to_owned(),
                    "id" => id.to_string(),
                    "type_symbol" => symbol(atom),
                    "label_atom_id" | "auth_atom_id" => name.clone(),
                    "label_alt_id" | "label_seq_id" => ".".to_owned(),
                    "label_comp_id" | "auth_comp_id" => comp.clone(),
                    "label_asym_id" => ids.new_asym.clone(),
                    "label_entity_id" => ids.new_entity.clone(),
                    "auth_asym_id" => ids.new_auth_asym.clone(),
                    "auth_seq_id" => auth_seq.clone(),
                    "cartn_x" => format!("{:.3}", posit.x),
                    "cartn_y" => format!("{:.3}", posit.y),
                    "cartn_z" => format!("{:.3}", posit.z),
                    "occupancy" => format!("{:.2}", atom.occupancy.unwrap_or(1.)),
                    "b_iso_or_equiv" => constant.unwrap_or_else(|| "0.00".to_owned()),
                    "pdbx_pdb_ins_code" | "pdbx_formal_charge" => "?".to_owned(),
                    _ => constant?,
                })
            });

            new_rows.push((None, row));
            atoms_out.push((i, id));
        }
    }

    if atoms_out.is_empty() {
        return Err(invalid("The ligand has no atoms to add"));
    }

    // Everything the ligand had when detached is still here, where it was, under the same IDs.
    let intact = !moved && matched.len() == o_rows.len();

    {
        let cat = doc
            .category_mut("_atom_site")
            .ok_or_else(|| invalid("Missing _atom_site"))?;
        let in_place = origin_site.is_some_and(|r| r.len_after == cat.len());
        if in_place {
            // Restored rows where they were, and any new ones (e.g. hydrogens) after them.
            new_rows.sort_by_key(|(i, _)| i.unwrap_or(usize::MAX));
        }

        let mut last = None;
        for (orig_i, row) in new_rows {
            let i = match (in_place, orig_i) {
                (true, Some(i)) => i,
                _ => match last {
                    Some(l) => l + 1,
                    None => before_water(cat, "label_comp_id"),
                },
            };
            cat.insert_row(i, row);
            last = Some(i);
        }
    }

    // The instance: its asym, and its entry in the non-polymer scheme.
    match origin {
        Some(o) => {
            for rec in o.removed.iter().rev() {
                if rec.category.eq_ignore_ascii_case("_atom_site")
                    || !is_instance_cat(&rec.category)
                {
                    continue;
                }
                let kind = IdKind::of(&rec.category);
                restore_rows(
                    doc,
                    rec,
                    same_protein,
                    |tags, row| rewrite_ids(tags, row, &ids, kind),
                    append,
                );
            }
        }
        None => {
            if let Some(cat) = doc.category_mut("_struct_asym") {
                let tags = cat.tags().to_vec();
                cat.push_row(row_from(&tags, |t| {
                    Some(match t {
                        "id" => ids.new_asym.clone(),
                        "pdbx_blank_pdb_chainid_flag" | "pdbx_modified" => "N".to_owned(),
                        "entity_id" => ids.new_entity.clone(),
                        _ => return None,
                    })
                }));
            }

            if let Some(cat) = doc.category_mut("_pdbx_nonpoly_scheme") {
                let tags = cat.tags().to_vec();
                let i = before_water(cat, "mon_id");
                cat.insert_row(
                    i,
                    row_from(&tags, |t| {
                        Some(match t {
                            "asym_id" => ids.new_asym.clone(),
                            "entity_id" => ids.new_entity.clone(),
                            "mon_id" | "pdb_mon_id" | "auth_mon_id" => comp.clone(),
                            "ndb_seq_num" => "1".to_owned(),
                            "pdb_seq_num" | "auth_seq_num" => auth_seq.clone(),
                            "pdb_strand_id" => ids.new_auth_asym.clone(),
                            "pdb_ins_code" => ".".to_owned(),
                            _ => return None,
                        })
                    }),
                );
            }
        }
    }

    // Its entity: an existing one if the file already has this component, or a new one.
    let (formula, weight) = formula_weight(atoms);
    let descrip = ligand_descrip(inp.lig);

    if new_entity {
        if let Some(o) = origin {
            for rec in o.descriptions.iter().rev() {
                if !is_entity_cat(&rec.category) {
                    continue;
                }
                let kind = IdKind::of(&rec.category);
                restore_rows(
                    doc,
                    rec,
                    same_protein,
                    |tags, row| {
                        rewrite_ids(tags, row, &ids, kind);
                        if kind == IdKind::Entity {
                            set_tag(row, tags, "pdbx_number_of_molecules", "1");
                        }
                    },
                    append,
                );
            }
        }

        let has_row = |doc: &CifDoc, cat: &str, tag: &str| {
            col_values(doc, cat, tag).contains(&ids.new_entity.as_str())
        };

        if doc.category("_entity").is_some() && !has_row(doc, "_entity", "id") {
            let cat = doc.category_mut("_entity").unwrap();
            let tags = cat.tags().to_vec();
            cat.push_row(row_from(&tags, |t| {
                Some(match t {
                    "id" => ids.new_entity.clone(),
                    "type" => "non-polymer".to_owned(),
                    "src_method" => "syn".to_owned(),
                    "pdbx_description" => descrip.clone(),
                    "formula_weight" => format!("{weight:.3}"),
                    "pdbx_number_of_molecules" => "1".to_owned(),
                    _ => return None,
                })
            }));
        }

        if doc.category("_pdbx_entity_nonpoly").is_some()
            && !has_row(doc, "_pdbx_entity_nonpoly", "entity_id")
        {
            let cat = doc.category_mut("_pdbx_entity_nonpoly").unwrap();
            let tags = cat.tags().to_vec();
            cat.push_row(row_from(&tags, |t| {
                Some(match t {
                    "entity_id" => ids.new_entity.clone(),
                    "name" => descrip.clone(),
                    "comp_id" => comp.clone(),
                    _ => return None,
                })
            }));
        }
    } else {
        adjust_entity_count(doc, &ids.new_entity, 1);
    }

    // Its chemical component, if the file describes components and doesn't have this one.
    let comp_described = col_values(doc, "_chem_comp", "id")
        .iter()
        .any(|c| c.eq_ignore_ascii_case(&comp));

    if doc.category("_chem_comp").is_some() && !comp_described {
        let from_origin = origin.filter(|o| {
            o.descriptions
                .iter()
                .any(|r| r.category.eq_ignore_ascii_case("_chem_comp"))
        });

        match from_origin {
            Some(o) => {
                for rec in o.descriptions.iter().rev() {
                    if !is_chem_comp_cat(&rec.category) {
                        continue;
                    }
                    let kind = IdKind::of(&rec.category);
                    let key = if kind == IdKind::Comp {
                        "id"
                    } else {
                        "comp_id"
                    };
                    restore_rows(
                        doc,
                        rec,
                        same_protein,
                        |tags, row| rewrite_ids(tags, row, &ids, kind),
                        |cat| sorted_index(cat, key, &comp),
                    );
                }
            }
            None => add_chem_comp(doc, inp, &comp, &descrip, &formula, weight),
        }
    }

    // Its connections, binding sites etc; only valid if it's back as it was.
    if let Some(o) = origin
        && same_protein
        && intact
        && ids.unchanged()
    {
        for rec in o.removed.iter().rev() {
            if !is_instance_cat(&rec.category) {
                restore_rows(doc, rec, true, |_, _| {}, append);
            }
        }
        recount_sites(
            doc,
            &col_values_of(&o.removed, "_struct_site_gen", "site_id"),
        );
    }

    let assemblies = match origin {
        Some(o) if same_protein => o.assemblies.clone(),
        _ => Vec::new(),
    };
    add_to_assemblies(
        doc,
        &ids.new_asym,
        &assemblies,
        nearest.as_ref().map(|n| n.0.as_str()),
    );

    let mut symbols: Vec<String> = atoms_out.iter().map(|(i, _)| symbol(&atoms[*i])).collect();
    symbols.dedup();
    add_atom_types(doc, &symbols);

    Ok(AttachOutput {
        atoms: atoms_out,
        label_asym_id: ids.new_asym,
        label_seq: origin
            .and_then(|o| o.label_seq_id.parse().ok())
            .unwrap_or(0),
        comp_id: comp,
    })
}

/// Describe a new chemical component: `_chem_comp`, and its atoms and bonds if the file lists
/// those for its other components.
fn add_chem_comp(
    doc: &mut CifDoc,
    inp: &AttachInput,
    comp: &str,
    descrip: &str,
    formula: &str,
    weight: f32,
) {
    let mol = &inp.lig.common;

    if let Some(cat) = doc.category_mut("_chem_comp") {
        let tags = cat.tags().to_vec();
        let i = sorted_index(cat, "id", comp);
        cat.insert_row(
            i,
            row_from(&tags, |t| {
                Some(match t {
                    "id" => comp.to_owned(),
                    "type" => "non-polymer".to_owned(),
                    "mon_nstd_flag" => ".".to_owned(),
                    "name" => descrip.to_owned(),
                    "formula" => formula.to_owned(),
                    "formula_weight" => format!("{weight:.3}"),
                    _ => return None,
                })
            }),
        );
    }

    let mut aromatic = vec![false; mol.atoms.len()];
    for bond in &mol.bonds {
        if bond.bond_type == BondType::Aromatic {
            for i in [bond.atom_0, bond.atom_1] {
                if let Some(a) = aromatic.get_mut(i) {
                    *a = true;
                }
            }
        }
    }
    let yn = |b: bool| if b { "Y" } else { "N" }.to_owned();

    let next_ordinal = |cat: &CifCategory| {
        cat.col("pdbx_ordinal")
            .and_then(|c| {
                cat.rows()
                    .iter()
                    .filter_map(|r| r.get(c).parse::<u32>().ok())
                    .max()
            })
            .unwrap_or(0)
            + 1
    };

    if let Some(cat) = doc.category_mut("_chem_comp_atom") {
        let tags = cat.tags().to_vec();
        let mut ordinal = next_ordinal(cat);
        let mut i = sorted_index(cat, "comp_id", comp);

        for (a, atom) in mol.atoms.iter().enumerate() {
            cat.insert_row(
                i,
                row_from(&tags, |t| {
                    Some(match t {
                        "comp_id" => comp.to_owned(),
                        "atom_id" => inp.names[a].clone(),
                        "type_symbol" => atom.element.to_letter().to_uppercase(),
                        "pdbx_aromatic_flag" => yn(aromatic[a]),
                        "pdbx_stereo_config" => "N".to_owned(),
                        "pdbx_ordinal" => ordinal.to_string(),
                        _ => return None,
                    })
                }),
            );
            i += 1;
            ordinal += 1;
        }
    }

    if let Some(cat) = doc.category_mut("_chem_comp_bond") {
        let tags = cat.tags().to_vec();
        let mut ordinal = next_ordinal(cat);
        let mut i = sorted_index(cat, "comp_id", comp);

        for bond in &mol.bonds {
            let (Some(n0), Some(n1)) = (inp.names.get(bond.atom_0), inp.names.get(bond.atom_1))
            else {
                continue;
            };
            let (order, is_aromatic) = bond_order(bond.bond_type);

            cat.insert_row(
                i,
                row_from(&tags, |t| {
                    Some(match t {
                        "comp_id" => comp.to_owned(),
                        "atom_id_1" => n0.clone(),
                        "atom_id_2" => n1.clone(),
                        "value_order" => order.to_owned(),
                        "pdbx_aromatic_flag" => yn(is_aromatic),
                        "pdbx_stereo_config" => "N".to_owned(),
                        "pdbx_ordinal" => ordinal.to_string(),
                        _ => return None,
                    })
                }),
            );
            i += 1;
            ordinal += 1;
        }
    }
}

impl MoleculePeptide {
    /// Whether a residue is a ligand, ion, cofactor, or other hetero group: made of hetero atoms,
    /// and not an amino acid or water.
    pub fn is_ligand_res(&self, res_i: usize) -> bool {
        let Some(res) = self.residues.get(res_i) else {
            return false;
        };

        matches!(res.res_type, ResidueType::Other(_))
            && !res.atoms.is_empty()
            && res
                .atoms
                .iter()
                .all(|&i| self.common.atoms.get(i).is_some_and(|a| a.hetero))
    }

    /// Indices of residues that are ligands, ions, cofactors etc. See [`Self::is_ligand_res`].
    pub fn ligand_residues(&self) -> Vec<usize> {
        (0..self.residues.len())
            .filter(|&i| self.is_ligand_res(i))
            .collect()
    }

    /// Remove a ligand (or other hetero residue) from this protein, and from its source mmCIF,
    /// along with the records describing it. Returns those records, if there's an mmCIF.
    pub fn remove_het_residue(&mut self, res_i: usize) -> io::Result<Option<LigandCifOrigin>> {
        if !self.is_ligand_res(res_i) {
            return Err(invalid(
                "This residue isn't a ligand or other hetero group, so can't be removed",
            ));
        }
        let res = &self.residues[res_i];

        // Edit the mmCIF first, so if that fails, we've changed nothing.
        let cif = match &self.source_cif {
            Some(text) => {
                let mut doc = CifDoc::new(text)?;

                let sns: HashSet<u32> = res
                    .atoms
                    .iter()
                    .map(|&i| self.common.atoms[i].serial_number)
                    .collect();
                let chain_id = res
                    .atoms
                    .first()
                    .and_then(|&i| self.common.atoms[i].chain)
                    .and_then(|c| self.chains.get(c))
                    .map(|c| c.id.clone());

                let inst =
                    find_instance(&doc, &sns, chain_id.as_deref(), &res.res_type.to_string())?;
                let origin = remove_instance(&mut doc, &inst, &self.common.ident);

                Some((origin, doc))
            }
            None => None,
        };

        self.remove_residue_in_place(res_i);

        Ok(cif.map(|(origin, doc)| {
            sync_metadata(&mut self.common.metadata, &doc);
            self.source_cif = Some(doc.to_text());
            origin
        }))
    }

    /// Remove a ligand (or other hetero residue) from this protein and its mmCIF, and return it as
    /// a standalone molecule, in place. It carries its mmCIF records, so re-attaching it with
    /// [`Self::attach_ligand`] restores them.
    pub fn detach_het_residue(&mut self, res_i: usize) -> io::Result<MoleculeSmall> {
        self.detach_het_residue_with_fragments(res_i, true)
    }

    /// Detach the entire residue, optionally discarding all but its largest bonded component
    /// from the returned ligand. Component size is ranked by heavy-atom count.
    pub fn detach_het_residue_with_fragments(
        &mut self,
        res_i: usize,
        include_disconnected: bool,
    ) -> io::Result<MoleculeSmall> {
        let mut res = self
            .residues
            .get(res_i)
            .cloned()
            .ok_or_else(|| invalid("Residue index out of range"))?;

        if !include_disconnected {
            res.atoms = self.common.largest_connected_component(&res.atoms);
        }

        let mut result = MoleculeSmall::from_res(&res, &self.common.atoms, &self.common.bonds);

        // `from_res` centers the ligand on the origin; keep it where it is instead.
        for (lig_i, &pep_i) in res.atoms.iter().enumerate() {
            let posit = self
                .common
                .atom_posits
                .get(pep_i)
                .copied()
                .unwrap_or(self.common.atoms[pep_i].posit);

            let atom = &mut result.common.atoms[lig_i];
            atom.posit = posit;
            if let Some(AtomTypeInRes::Hetero(name)) = &mut atom.type_in_res {
                *name = unquote(name).to_owned();
            }
            result.common.atom_posits[lig_i] = posit;
        }

        result.cif_origin = self.remove_het_residue(res_i)?;

        Ok(result)
    }

    /// Add a ligand to this protein as a hetero residue, at its current position, and to the
    /// protein's source mmCIF as a non-polymer instance, with entity and chemical component
    /// records. `comp_id` is its residue name, e.g. "ATP"; sharing one already in the file makes
    /// it another instance of that component.
    ///
    /// If the ligand was detached from a protein, this restores its mmCIF records; see
    /// [`LigandCifOrigin`].
    ///
    /// Hydrogens are included only if the mmCIF has them already, as X-ray structures usually
    /// don't. Returns the new residue's index.
    pub fn attach_ligand(&mut self, lig: &MoleculeSmall, comp_id: &str) -> io::Result<usize> {
        let comp_id = normalize_comp_id(comp_id)?;
        let common = &lig.common;
        if common.atoms.is_empty() {
            return Err(invalid("The ligand has no atoms"));
        }

        let world: Vec<Vec3> = if common.atom_posits.len() == common.atoms.len() {
            common.atom_posits.clone()
        } else {
            common.atoms.iter().map(|a| a.posit).collect()
        };
        // The mmCIF, and our atoms' local positions, are in the protein's original frame.
        let frame = RigidFrame::new(&self.common);
        let local: Vec<Vec3> = world.iter().map(|p| frame.to_local(*p)).collect();

        let names = ligand_atom_names(common);
        let used_sns: HashSet<u32> = self.common.atoms.iter().map(|a| a.serial_number).collect();

        let (added, chain_id, res_sn, comp_id, doc) = match &self.source_cif {
            Some(text) => {
                let mut doc = CifDoc::new(text)?;
                let include_h = doc_has_hydrogen(&doc);
                let out = attach_to_doc(
                    &mut doc,
                    &AttachInput {
                        lig,
                        names: &names,
                        posits: &local,
                        comp_id: &comp_id,
                        include_h,
                        target_ident: &self.common.ident,
                        used_sns: &used_sns,
                    },
                )?;

                (
                    out.atoms,
                    out.label_asym_id,
                    out.label_seq,
                    out.comp_id,
                    Some(doc),
                )
            }
            None => {
                let mut sn = used_sns.iter().max().copied().unwrap_or(0);
                let atoms = (0..common.atoms.len())
                    .map(|i| {
                        sn += 1;
                        (i, sn)
                    })
                    .collect();
                let chains: HashSet<String> = self.chains.iter().map(|c| c.id.clone()).collect();

                (atoms, next_asym_id(&chains), 0, comp_id, None)
            }
        };

        let mut index = HashMap::new();
        let mut atoms = Vec::with_capacity(added.len());
        let mut posits = Vec::with_capacity(added.len());

        for (k, &(i, sn)) in added.iter().enumerate() {
            let src = &common.atoms[i];
            let type_in_res = AtomTypeInRes::Hetero(names[i].clone());

            // As the mmCIF parser would create it.
            atoms.push(Atom {
                serial_number: sn,
                posit: local[i],
                element: src.element,
                role: Some(AtomRole::from_type_in_res(&type_in_res)),
                type_in_res: Some(type_in_res),
                hetero: true,
                occupancy: src.occupancy,
                ..Default::default()
            });
            posits.push(world[i]);
            index.insert(i, k);
        }

        let bonds = common
            .bonds
            .iter()
            .filter_map(|b| Some((*index.get(&b.atom_0)?, *index.get(&b.atom_1)?, b.bond_type)))
            .collect();

        let res_i = self.append_residue_in_place(
            atoms,
            posits,
            bonds,
            ResidueType::from_str(&comp_id),
            res_sn,
            &chain_id,
        );

        if let Some(doc) = doc {
            sync_metadata(&mut self.common.metadata, &doc);
            self.source_cif = Some(doc.to_text());
        }

        Ok(res_i)
    }

    /// A residue name (chemical component ID) to suggest for attaching a ligand: its own if it
    /// has one, or a placeholder like "LIG" not used by this protein.
    pub fn suggest_comp_id(&self, lig: &MoleculeSmall) -> String {
        if let Some(o) = &lig.cif_origin
            && o.comp_ids.len() == 1
        {
            return o.comp_ids[0].clone();
        }

        let valid = |s: &str| normalize_comp_id(s).is_ok();
        for ident in &lig.idents {
            if let MolIdent::PdbeAmber(code) = ident
                && valid(code)
            {
                return code.to_ascii_uppercase();
            }
        }

        let used: HashSet<String> = self
            .residues
            .iter()
            .filter_map(|r| match &r.res_type {
                ResidueType::Other(name) => Some(name.to_ascii_uppercase()),
                _ => None,
            })
            .collect();

        let ident = lig.common.ident.trim();
        if ident.len() <= 3 && valid(ident) && !used.contains(&ident.to_ascii_uppercase()) {
            return ident.to_ascii_uppercase();
        }

        std::iter::once("LIG".to_owned())
            .chain((1..=9).map(|n| format!("LG{n}")))
            .chain((10..=99).map(|n| format!("L{n}")))
            .find(|c| !used.contains(c))
            .unwrap_or_else(|| "LIG".to_owned())
    }

    /// Remove a residue and its atoms, re-indexing everything that refers to atoms, residues,
    /// or chains by index.
    fn remove_residue_in_place(&mut self, res_i: usize) {
        let removed: HashSet<usize> = self.residues[res_i].atoms.iter().copied().collect();
        let removed_sns: HashSet<u32> = removed
            .iter()
            .filter_map(|&i| self.common.atoms.get(i))
            .map(|a| a.serial_number)
            .collect();
        let res_sn = self.residues[res_i].serial_number;

        let n = self.common.atoms.len();
        let mut atom_map = vec![None; n];
        let mut next = 0;
        for (i, new_i) in atom_map.iter_mut().enumerate() {
            if !removed.contains(&i) {
                *new_i = Some(next);
                next += 1;
            }
        }
        let map_atom = |i: usize| atom_map.get(i).copied().flatten();

        let mut i = 0;
        self.common.atoms.retain(|_| {
            i += 1;
            atom_map[i - 1].is_some()
        });
        if self.common.atom_posits.len() == n {
            let mut i = 0;
            self.common.atom_posits.retain(|_| {
                i += 1;
                atom_map[i - 1].is_some()
            });
        } else {
            self.common.reset_posits();
        }

        self.common
            .bonds
            .retain_mut(|b| match (map_atom(b.atom_0), map_atom(b.atom_1)) {
                (Some(a0), Some(a1)) => {
                    b.atom_0 = a0;
                    b.atom_1 = a1;
                    true
                }
                _ => false,
            });

        self.bonds_hydrogen.retain_mut(|h| {
            match (
                map_atom(h.donor),
                map_atom(h.acceptor),
                map_atom(h.hydrogen),
            ) {
                (Some(d), Some(a), Some(hy)) => {
                    h.donor = d;
                    h.acceptor = a;
                    h.hydrogen = hy;
                    true
                }
                _ => false,
            }
        });

        self.residues.remove(res_i);
        let map_res = |r: usize| match r.cmp(&res_i) {
            std::cmp::Ordering::Less => Some(r),
            std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => Some(r - 1),
        };
        for res in &mut self.residues {
            res.atoms = res.atoms.iter().filter_map(|&a| map_atom(a)).collect();
        }

        // Chains that held only this residue go with it.
        let mut chain_map = Vec::with_capacity(self.chains.len());
        let mut kept = 0;
        for chain in &mut self.chains {
            let had = chain.atom_sns.iter().any(|sn| removed_sns.contains(sn));

            chain.atoms = chain.atoms.iter().filter_map(|&a| map_atom(a)).collect();
            chain.atom_sns.retain(|sn| !removed_sns.contains(sn));
            chain.residues = chain.residues.iter().filter_map(|&r| map_res(r)).collect();

            if had
                && !chain.residues.iter().any(|&r| {
                    self.residues
                        .get(r)
                        .is_some_and(|r| r.serial_number == res_sn)
                })
            {
                chain.residue_sns.retain(|&sn| sn != res_sn);
            }

            if had && chain.atoms.is_empty() {
                chain_map.push(None);
            } else {
                chain_map.push(Some(kept));
                kept += 1;
            }
        }
        let mut c = 0;
        self.chains.retain(|_| {
            c += 1;
            chain_map[c - 1].is_some()
        });

        for atom in &mut self.common.atoms {
            atom.residue = atom.residue.and_then(map_res);
            atom.chain = atom.chain.and_then(|c| chain_map.get(c).copied().flatten());
        }

        if let Some(filtered) = &mut self.atoms_filtered_to_disp {
            *filtered = filtered.iter().filter_map(|&i| map_atom(i)).collect();
        }

        self.after_topology_change();
    }

    /// Add a hetero residue. Bonds are (atom index, atom index) within `atoms`.
    fn append_residue_in_place(
        &mut self,
        mut atoms: Vec<Atom>,
        posits: Vec<Vec3>,
        bonds: Vec<(usize, usize, BondType)>,
        res_type: ResidueType,
        res_sn: u32,
        chain_id: &str,
    ) -> usize {
        if self.common.atom_posits.len() != self.common.atoms.len() {
            self.common.reset_posits();
        }

        let offset = self.common.atoms.len();
        let res_i = self.residues.len();
        let chain_i = match self.chains.iter().position(|c| c.id == chain_id) {
            Some(i) => i,
            None => {
                self.chains.push(Chain {
                    id: chain_id.to_owned(),
                    residue_sns: Vec::new(),
                    residues: Vec::new(),
                    atom_sns: Vec::new(),
                    atoms: Vec::new(),
                    visible: true,
                });
                self.chains.len() - 1
            }
        };

        let sns: Vec<u32> = atoms.iter().map(|a| a.serial_number).collect();
        let indices: Vec<usize> = (offset..offset + atoms.len()).collect();
        for atom in &mut atoms {
            atom.residue = Some(res_i);
            atom.chain = Some(chain_i);
        }

        self.common.atoms.extend(atoms);
        self.common.atom_posits.extend(posits);

        for (a0, a1, bond_type) in bonds {
            self.common.bonds.push(Bond {
                bond_type,
                atom_0_sn: sns[a0],
                atom_1_sn: sns[a1],
                atom_0: offset + a0,
                atom_1: offset + a1,
                is_backbone: false,
            });
        }

        self.residues.push(Residue {
            serial_number: res_sn,
            res_type,
            atom_sns: sns.clone(),
            atoms: indices.clone(),
            dihedral: None,
            end: ResidueEnd::Hetero,
        });

        let chain = &mut self.chains[chain_i];
        chain.atoms.extend(&indices);
        chain.atom_sns.extend(&sns);
        chain.residues.push(res_i);
        if !chain.residue_sns.contains(&res_sn) {
            chain.residue_sns.push(res_sn);
        }

        self.after_topology_change();
        res_i
    }

    fn after_topology_change(&mut self) {
        self.update_het_residues();
        self.common.build_adjacency_list();
        self.common.update_next_sn();
        self.common.entity_i_range = None;

        let (center, size) = mol_center_size(&self.common.atoms);
        self.center = center;
        self.size = size;
    }
}
