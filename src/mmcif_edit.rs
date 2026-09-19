//! Text-preserving reading and editing of mmCIF files.
//!
//! [`bio_files::MmCif`] parses what we build molecules from, and drops the rest. That's what we
//! want when loading, but not when editing: an entry from the RCSB PDB carries dozens of other
//! categories (entities, chemical components, connections, binding sites, validation reports...),
//! which we'd lose by writing out only what we parsed.
//!
//! [`CifDoc`] instead keeps every category, along with the source text of each category and row.
//! Those we don't change are written back out exactly as read. Values we change are replaced in
//! place where possible, so the file's column alignment survives too.

use std::{fmt::Write, io, io::ErrorKind};

#[derive(Clone, Copy, PartialEq, Debug)]
enum TokKind {
    Unquoted,
    Quoted,
    /// A multi-line value, delimited by semicolons at the start of lines.
    TextField,
}

#[derive(Debug)]
struct Token {
    /// Byte range in the source text, including any quotes or semicolons.
    start: usize,
    end: usize,
    kind: TokKind,
    /// With quotes and text-field delimiters removed.
    value: String,
}

impl Token {
    fn is_tag(&self) -> bool {
        self.kind == TokKind::Unquoted && self.value.starts_with('_')
    }

    /// `loop_`, `data_`, `save_`, `global_` and `stop_` are reserved words, not values.
    fn is_reserved(&self) -> bool {
        if self.kind != TokKind::Unquoted {
            return false;
        }
        let v = self.value.to_ascii_lowercase();

        v == "loop_"
            || v.starts_with("data_")
            || v.starts_with("save_")
            || v == "global_"
            || v == "stop_"
    }

    fn is_value(&self) -> bool {
        !self.is_tag() && !self.is_reserved()
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Split mmCIF text into tokens, per the CIF 1.1 syntax: whitespace-delimited, with single- or
/// double-quoted strings, semicolon-delimited text fields, and `#` comments.
fn tokenize(text: &str) -> Vec<Token> {
    let b = text.as_bytes();
    let n = b.len();
    let mut result = Vec::new();

    let mut i = 0;
    // Text fields open with a semicolon at the start of a line.
    let mut line_start = true;

    while i < n {
        let c = b[i];

        if c == b'\n' {
            line_start = true;
            i += 1;
            continue;
        }

        if line_start && c == b';' {
            // The field ends at the next line that starts with a semicolon.
            let mut j = i + 1;
            let close = loop {
                match text[j..].find('\n') {
                    Some(k) => {
                        let next = j + k + 1;
                        if next < n && b[next] == b';' {
                            break Some(next);
                        }
                        j = next;
                    }
                    None => break None,
                }
            };

            let (value_end, end) = match close {
                // Exclude the newline before the closing semicolon.
                Some(close) => (close - 1, close + 1),
                None => (n, n),
            };
            let value = text[i + 1..value_end.max(i + 1)].trim_end_matches('\r');

            result.push(Token {
                start: i,
                end,
                kind: TokKind::TextField,
                value: value.to_owned(),
            });

            i = end;
            line_start = false;
            continue;
        }

        line_start = false;

        if is_ws(c) {
            i += 1;
            continue;
        }

        if c == b'#' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        if c == b'\'' || c == b'"' {
            // A quote only closes the string if followed by whitespace; e.g. `"O5'"`.
            let mut j = i + 1;
            while j < n && b[j] != b'\n' && !(b[j] == c && (j + 1 == n || is_ws(b[j + 1]))) {
                j += 1;
            }

            let end = if j < n && b[j] == c { j + 1 } else { j };

            result.push(Token {
                start: i,
                end,
                kind: TokKind::Quoted,
                value: text[i + 1..j].to_owned(),
            });

            i = end;
            continue;
        }

        let mut j = i;
        while j < n && !is_ws(b[j]) {
            j += 1;
        }

        result.push(Token {
            start: i,
            end: j,
            kind: TokKind::Unquoted,
            value: text[i..j].to_owned(),
        });

        i = j;
    }

    result
}

/// Format a value for writing, quoting it if required.
pub fn format_value(v: &str) -> String {
    if v.is_empty() {
        return "''".to_owned();
    }

    let lower = v.to_ascii_lowercase();
    let needs_quotes = v
        .chars()
        .any(|c| c.is_whitespace() || c == '\'' || c == '"')
        || v.starts_with(['_', '#', '$', ';', '[', ']'])
        || lower.starts_with("data_")
        || lower.starts_with("save_")
        || lower == "loop_"
        || lower == "global_"
        || lower == "stop_";

    if !needs_quotes {
        v.to_owned()
    } else if !v.contains('\'') {
        format!("'{v}'")
    } else if !v.contains('"') {
        format!("\"{v}\"")
    } else if !v.contains("' ") {
        // Quote characters inside a quoted string are fine unless followed by whitespace.
        format!("'{v}'")
    } else {
        format!("\"{v}\"")
    }
}

/// Split e.g. `_atom_site.Cartn_x` into `("_atom_site", "Cartn_x")`.
fn split_tag(tag: &str) -> (&str, &str) {
    tag.split_once('.').unwrap_or((tag, ""))
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg.into())
}

/// One row of a category: a row of a loop, or the single row of a set of key-value items.
#[derive(Clone, Debug)]
pub struct CifRow {
    /// In tag order, with quotes removed. `?` and `.` are mmCIF's markers for unknown and
    /// inapplicable values.
    values: Vec<String>,
    /// The row's source text (whole lines), while it's still valid to write out.
    raw: Option<String>,
    /// Where each value's token sits in `raw`. `None` for values we can't edit in place.
    spans: Vec<Option<(usize, usize)>>,
}

impl CifRow {
    pub fn new(values: Vec<String>) -> Self {
        let spans = vec![None; values.len()];
        Self {
            values,
            raw: None,
            spans,
        }
    }

    fn from_tokens(tokens: &[&Token]) -> Self {
        Self {
            values: tokens.iter().map(|t| t.value.clone()).collect(),
            raw: None,
            // Absolute byte positions for now; made relative to `raw` once we know it's valid.
            spans: tokens
                .iter()
                .map(|t| (t.kind != TokKind::TextField).then_some((t.start, t.end)))
                .collect(),
        }
    }

    /// Set the source text to lines `start..end` (bytes) of `text`, converting spans to be
    /// relative to it.
    fn set_raw(&mut self, text: &str, start: usize, end: usize) {
        self.raw = Some(text[start..end].to_owned());
        for span in &mut self.spans {
            *span =
                span.and_then(|(s, e)| (s >= start && e <= end).then(|| (s - start, e - start)));
        }
    }

    fn clear_raw(&mut self) {
        self.raw = None;
        self.spans.fill(None);
    }

    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// The value at a column index. Empty if out of range.
    pub fn get(&self, col: usize) -> &str {
        self.values.get(col).map(String::as_str).unwrap_or("")
    }

    /// Set the value at a column index. Where we can, this edits the row's source text in place,
    /// absorbing any change in width into the whitespace after it so later columns stay aligned.
    pub fn set(&mut self, col: usize, value: &str) {
        if col >= self.values.len() || self.values[col] == value {
            return;
        }
        self.values[col] = value.to_owned();

        let Some(raw) = &mut self.raw else {
            return;
        };
        let Some((s, e)) = self.spans[col] else {
            self.clear_raw();
            return;
        };
        if value.contains('\n') {
            self.clear_raw();
            return;
        }

        let token = format_value(value);
        let (old_len, new_len) = (e - s, token.len());

        let (replace_end, replacement) = if new_len <= old_len {
            (e, format!("{token}{}", " ".repeat(old_len - new_len)))
        } else {
            let after = &raw.as_bytes()[e..];
            let ws_after = after.iter().take_while(|b| **b == b' ').count();
            let at_line_end = after
                .get(ws_after)
                .is_none_or(|b| *b == b'\n' || *b == b'\r');
            // Keep at least one space between values.
            let keep = if at_line_end { 0 } else { 1 };
            let absorb = (new_len - old_len).min(ws_after.saturating_sub(keep));

            (e + absorb, token)
        };

        let shift = replacement.len() as isize - (replace_end - s) as isize;
        raw.replace_range(s..replace_end, &replacement);

        for (i, span) in self.spans.iter_mut().enumerate() {
            if i == col {
                *span = Some((s, s + new_len));
            } else if let Some((s_, e_)) = span
                && *s_ >= replace_end
            {
                *s_ = (*s_ as isize + shift) as usize;
                *e_ = (*e_ as isize + shift) as usize;
            }
        }
    }
}

/// One mmCIF category, e.g. `_atom_site`: either a loop, or a set of key-value items.
#[derive(Clone, Debug)]
pub struct CifCategory {
    /// E.g. `_atom_site`.
    name: String,
    /// Item names, without the category prefix. E.g. `Cartn_x`.
    tags: Vec<String>,
    rows: Vec<CifRow>,
    is_loop: bool,
    /// The `loop_` line, and tag lines, as read.
    header_raw: Option<String>,
    /// Inclusive source line range. `None` for categories we've added.
    span: Option<(usize, usize)>,
    modified: bool,
}

impl CifCategory {
    /// A new, empty category, written as a loop.
    pub fn new_loop(name: &str, tags: Vec<String>) -> Self {
        Self {
            name: name.to_owned(),
            tags,
            rows: Vec::new(),
            is_loop: true,
            header_raw: None,
            span: None,
            modified: true,
        }
    }

    /// Re-create a category from parts of one read earlier; e.g. to restore rows removed from it.
    pub(crate) fn from_parts(
        name: &str,
        tags: Vec<String>,
        is_loop: bool,
        header_raw: Option<String>,
        rows: Vec<CifRow>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            tags,
            rows,
            is_loop: is_loop || header_raw.is_some(),
            header_raw,
            span: None,
            modified: true,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn rows(&self) -> &[CifRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn is_loop(&self) -> bool {
        self.is_loop
    }

    pub(crate) fn header_raw(&self) -> Option<&String> {
        self.header_raw.as_ref()
    }

    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// The index of a tag (item name). Case-insensitive, as in mmCIF.
    pub fn col(&self, tag: &str) -> Option<usize> {
        self.tags.iter().position(|t| t.eq_ignore_ascii_case(tag))
    }

    pub fn get(&self, row: usize, tag: &str) -> Option<&str> {
        let col = self.col(tag)?;
        Some(self.rows.get(row)?.get(col))
    }

    /// Set a value by row index and tag. Does nothing if the tag isn't present.
    pub fn set(&mut self, row: usize, tag: &str, value: &str) {
        if let Some(col) = self.col(tag) {
            self.set_col(row, col, value);
        }
    }

    pub fn set_col(&mut self, row: usize, col: usize, value: &str) {
        if let Some(r) = self.rows.get_mut(row)
            && r.get(col) != value
        {
            r.set(col, value);
            self.modified = true;
        }
    }

    /// Insert a row, with values in tag order. Adding a second row to key-value items converts
    /// them to a loop.
    pub fn insert_row(&mut self, i: usize, mut row: CifRow) {
        row.values.resize(self.tags.len(), "?".to_owned());
        row.spans.resize(self.tags.len(), None);

        if !self.rows.is_empty() {
            self.make_loop();
        }

        self.rows.insert(i.min(self.rows.len()), row);
        self.modified = true;
    }

    /// Write key-value items as a loop instead, e.g. before adding a second row.
    pub(crate) fn make_loop(&mut self) {
        if self.is_loop {
            return;
        }
        self.is_loop = true;
        self.header_raw = None;
        // The key-value form's source text holds tags; it can't be reused for a loop row.
        for r in &mut self.rows {
            r.clear_raw();
        }
        self.modified = true;
    }

    pub fn push_row(&mut self, row: CifRow) {
        self.insert_row(self.rows.len(), row);
    }

    /// Remove rows by index; returns them along with their indices.
    pub fn remove_rows(&mut self, indices: &[usize]) -> Vec<(usize, CifRow)> {
        let mut result = Vec::with_capacity(indices.len());
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        for &i in sorted.iter().rev() {
            if i < self.rows.len() {
                result.push((i, self.rows.remove(i)));
            }
        }
        result.reverse();

        if !result.is_empty() {
            self.modified = true;
        }
        result
    }

    /// Write the category out. `nl` is the line ending for lines we generate.
    fn to_text(&self, nl: &str) -> String {
        let mut out = String::new();
        let end_line = |out: &mut String| {
            if !out.ends_with('\n') {
                out.push_str(nl);
            }
        };

        if !self.is_loop && self.rows.len() == 1 {
            let row = &self.rows[0];
            if let Some(raw) = &row.raw {
                out.push_str(raw);
                end_line(&mut out);
                return out;
            }

            let width = self
                .tags
                .iter()
                .map(|t| self.name.len() + t.len() + 1)
                .max()
                .unwrap_or(0)
                + 3;

            for (tag, v) in self.tags.iter().zip(&row.values) {
                let full = format!("{}.{tag}", self.name);
                if v.contains('\n') {
                    let _ = write!(out, "{full}{nl};{v}{nl};{nl}");
                } else {
                    let _ = write!(out, "{full:<width$}{}{nl}", format_value(v));
                }
            }
            return out;
        }

        match &self.header_raw {
            Some(h) => {
                out.push_str(h);
                end_line(&mut out);
            }
            None => {
                let _ = write!(out, "loop_{nl}");
                for tag in &self.tags {
                    let _ = write!(out, "{}.{tag} {nl}", self.name);
                }
            }
        }

        // Pad values we format to the widest in each column, so they line up with the rest.
        let widths = if self.rows.iter().any(|r| r.raw.is_none()) {
            let mut w = vec![0; self.tags.len()];
            for row in &self.rows {
                for (i, v) in row.values.iter().enumerate() {
                    if !v.contains('\n') {
                        w[i] = w[i].max(format_value(v).chars().count());
                    }
                }
            }
            w
        } else {
            Vec::new()
        };

        for row in &self.rows {
            match &row.raw {
                Some(raw) => {
                    out.push_str(raw);
                    end_line(&mut out);
                }
                None => {
                    for (i, v) in row.values.iter().enumerate() {
                        if v.contains('\n') {
                            end_line(&mut out);
                            let _ = write!(out, ";{v}{nl};{nl}");
                            continue;
                        }

                        let token = format_value(v);
                        let pad = widths[i].saturating_sub(token.chars().count());
                        out.push_str(&token);
                        out.push_str(&" ".repeat(pad + 1));
                    }
                    end_line(&mut out);
                }
            }
        }

        out
    }
}

#[derive(Clone, Debug)]
enum Item {
    Category(CifCategory),
    /// A `data_` block header, or another reserved word.
    Other {
        line: usize,
        text: String,
    },
}

/// A parsed mmCIF document which retains its source text, for editing.
#[derive(Clone, Debug)]
pub struct CifDoc {
    text: String,
    line_starts: Vec<usize>,
    items: Vec<Item>,
    /// Whether each item has lines of its own, letting unmodified ones be written out from the
    /// source text. False for unusual layouts, e.g. a loop starting on the line another ends on.
    lossless: bool,
    /// The line separating categories, e.g. `# ` in files from the RCSB, if the file uses one.
    separator: Option<String>,
    /// The file's line ending, for lines we generate.
    newline: &'static str,
}

impl CifDoc {
    pub fn new(text: &str) -> io::Result<Self> {
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' && i + 1 < text.len() {
                line_starts.push(i + 1);
            }
        }
        let line_of = |byte: usize| match line_starts.binary_search(&byte) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let line_end = |line: usize| line_starts.get(line + 1).copied().unwrap_or(text.len());

        let tokens = tokenize(text);

        /// Line ranges of an item and its parts, for deciding which source text we can reuse.
        struct Lines {
            first: usize,
            last: usize,
            header: Option<(usize, usize)>,
            rows: Vec<(usize, usize)>,
        }

        let mut items = Vec::new();
        let mut lines = Vec::new();

        let mut i = 0;
        while i < tokens.len() {
            let tok = &tokens[i];
            let first = line_of(tok.start);

            if tok.kind == TokKind::Unquoted && tok.value.eq_ignore_ascii_case("loop_") {
                let mut j = i + 1;
                while j < tokens.len() && tokens[j].is_tag() {
                    j += 1;
                }
                let tag_tokens = &tokens[i + 1..j];
                if tag_tokens.is_empty() {
                    return Err(invalid("mmCIF loop_ with no tags"));
                }

                let name = split_tag(&tag_tokens[0].value).0.to_owned();
                let mut tags = Vec::with_capacity(tag_tokens.len());
                for t in tag_tokens {
                    let (cat, item) = split_tag(&t.value);
                    if !cat.eq_ignore_ascii_case(&name) {
                        return Err(invalid(format!(
                            "mmCIF loop mixes categories {name} and {cat}"
                        )));
                    }
                    tags.push(item.to_owned());
                }
                let header_last = line_of(tag_tokens[tag_tokens.len() - 1].end - 1);

                let values_start = j;
                while j < tokens.len() && tokens[j].is_value() {
                    j += 1;
                }
                let values: Vec<&Token> = tokens[values_start..j].iter().collect();
                if values.len() % tags.len() != 0 {
                    return Err(invalid(format!(
                        "mmCIF loop {name} has {} values for {} tags",
                        values.len(),
                        tags.len()
                    )));
                }

                let mut rows = Vec::with_capacity(values.len() / tags.len());
                let mut row_lines = Vec::with_capacity(rows.capacity());
                for chunk in values.chunks(tags.len()) {
                    row_lines.push((
                        line_of(chunk[0].start),
                        line_of(chunk[chunk.len() - 1].end - 1),
                    ));
                    rows.push(CifRow::from_tokens(chunk));
                }
                let last = row_lines.last().map(|r| r.1).unwrap_or(header_last);

                items.push(Item::Category(CifCategory {
                    name,
                    tags,
                    rows,
                    is_loop: true,
                    header_raw: None,
                    span: Some((first, last)),
                    modified: false,
                }));
                lines.push(Lines {
                    first,
                    last,
                    header: Some((first, header_last)),
                    rows: row_lines,
                });

                i = j;
            } else if tok.is_tag() {
                // Key-value items. Consecutive ones in the same category form one row.
                let name = split_tag(&tok.value).0.to_owned();
                let mut tags = Vec::new();
                let mut values = Vec::new();

                let mut j = i;
                while j + 1 < tokens.len()
                    && tokens[j].is_tag()
                    && split_tag(&tokens[j].value).0.eq_ignore_ascii_case(&name)
                    && tokens[j + 1].is_value()
                {
                    tags.push(split_tag(&tokens[j].value).1.to_owned());
                    values.push(&tokens[j + 1]);
                    j += 2;
                }
                if tags.is_empty() {
                    return Err(invalid(format!("mmCIF item {} has no value", tok.value)));
                }
                let last = line_of(tokens[j - 1].end - 1);

                items.push(Item::Category(CifCategory {
                    name,
                    tags,
                    rows: vec![CifRow::from_tokens(&values)],
                    is_loop: false,
                    header_raw: None,
                    span: Some((first, last)),
                    modified: false,
                }));
                lines.push(Lines {
                    first,
                    last,
                    header: None,
                    rows: vec![(first, last)],
                });

                i = j;
            } else if tok.is_reserved() {
                items.push(Item::Other {
                    line: first,
                    text: tok.value.clone(),
                });
                lines.push(Lines {
                    first,
                    last: first,
                    header: None,
                    rows: Vec::new(),
                });

                i += 1;
            } else {
                return Err(invalid(format!(
                    "Unexpected mmCIF value outside of a loop: {}",
                    tok.value
                )));
            }
        }

        // Now that we know where each item starts and ends, keep the source text of those items,
        // headers and rows that don't share lines with anything else.
        let mut lossless = true;
        for k in 0..items.len() {
            let l = &lines[k];
            let prev_last = k.checked_sub(1).map(|p| lines[p].last);
            let next_first = lines.get(k + 1).map(|n| n.first);

            let exclusive =
                prev_last.is_none_or(|p| l.first > p) && next_first.is_none_or(|n| l.last < n);
            lossless &= exclusive;

            let Item::Category(cat) = &mut items[k] else {
                continue;
            };

            if cat.is_loop {
                let (h_first, h_last) = l.header.unwrap_or((l.first, l.first));
                let after_header = l.rows.first().map(|r| r.0).or(next_first);
                if prev_last.is_none_or(|p| h_first > p) && after_header.is_none_or(|a| h_last < a)
                {
                    cat.header_raw = Some(text[line_starts[h_first]..line_end(h_last)].to_owned());
                }

                for r in 0..cat.rows.len() {
                    let (r_first, r_last) = l.rows[r];
                    let before = if r == 0 { h_last } else { l.rows[r - 1].1 };
                    let after = l.rows.get(r + 1).map(|x| x.0).or(next_first);

                    if r_first > before && after.is_none_or(|a| r_last < a) {
                        cat.rows[r].set_raw(text, line_starts[r_first], line_end(r_last));
                    } else {
                        cat.rows[r].clear_raw();
                    }
                }
            } else if exclusive {
                cat.rows[0].set_raw(text, line_starts[l.first], line_end(l.last));
            } else {
                cat.rows[0].clear_raw();
            }
        }

        Ok(Self {
            text: text.to_owned(),
            separator: text
                .split_inclusive('\n')
                .find(|l| l.trim() == "#")
                .map(|l| l.trim_end_matches(['\r', '\n']).to_owned()),
            newline: if text.contains("\r\n") { "\r\n" } else { "\n" },
            line_starts,
            items,
            lossless,
        })
    }

    fn line(&self, i: usize) -> &str {
        let end = self
            .line_starts
            .get(i + 1)
            .copied()
            .unwrap_or(self.text.len());
        &self.text[self.line_starts[i]..end]
    }

    pub fn categories(&self) -> impl Iterator<Item = &CifCategory> {
        self.items.iter().filter_map(|item| match item {
            Item::Category(c) => Some(c),
            _ => None,
        })
    }

    /// The first category with this name, e.g. `_atom_site`. May be empty, if we've removed all
    /// of its rows.
    pub fn category(&self, name: &str) -> Option<&CifCategory> {
        self.categories()
            .find(|c| c.name.eq_ignore_ascii_case(name))
    }

    pub fn category_mut(&mut self, name: &str) -> Option<&mut CifCategory> {
        self.items.iter_mut().find_map(|item| match item {
            Item::Category(c) if c.name.eq_ignore_ascii_case(name) => Some(c),
            _ => None,
        })
    }

    /// Whether a category exists, and has rows.
    pub fn has(&self, name: &str) -> bool {
        self.category(name).is_some_and(|c| !c.is_empty())
    }

    /// The name of the nearest category before this one with rows.
    pub fn category_before(&self, name: &str) -> Option<String> {
        let i = self.items.iter().position(
            |item| matches!(item, Item::Category(c) if c.name.eq_ignore_ascii_case(name)),
        )?;

        self.items[..i].iter().rev().find_map(|item| match item {
            Item::Category(c) if !c.is_empty() => Some(c.name.clone()),
            _ => None,
        })
    }

    /// Add a category after the one named, or at the end if it's absent.
    pub fn insert_category(&mut self, after: Option<&str>, cat: CifCategory) {
        let i = after
            .and_then(|a| {
                self.items.iter().position(
                    |item| matches!(item, Item::Category(c) if c.name.eq_ignore_ascii_case(a)),
                )
            })
            .map(|i| i + 1)
            .unwrap_or(self.items.len());

        self.items.insert(i, Item::Category(cat));
    }

    /// Write the document out. Unmodified categories and rows are written exactly as read.
    pub fn to_text(&self) -> String {
        if !self.lossless {
            return self.to_text_rebuilt();
        }

        let mut out = String::with_capacity(self.text.len() + 4_096);
        let mut next_line = 0;
        // Set after dropping a category that's become empty, to drop its trailing `#` separator
        // too.
        let mut skip_sep = false;

        for item in &self.items {
            let span = match item {
                Item::Category(c) => c.span,
                Item::Other { line, .. } => Some((*line, *line)),
            };

            let Some((first, last)) = span else {
                // A category we've added.
                let Item::Category(cat) = item else {
                    continue;
                };
                if cat.is_empty() {
                    continue;
                }
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push_str(self.newline);
                }
                // If the category before this one was dropped, its separator is still pending
                // and serves for this one.
                if let Some(sep) = &self.separator
                    && !std::mem::take(&mut skip_sep)
                {
                    out.push_str(sep);
                    out.push_str(self.newline);
                }
                out.push_str(&cat.to_text(self.newline));
                continue;
            };

            for l in next_line..first {
                let line = self.line(l);
                if std::mem::take(&mut skip_sep) && line.trim() == "#" {
                    continue;
                }
                out.push_str(line);
            }
            skip_sep = false;

            match item {
                Item::Category(cat) if cat.modified => {
                    if cat.is_empty() {
                        skip_sep = true;
                    } else {
                        out.push_str(&cat.to_text(self.newline));
                    }
                }
                _ => {
                    for l in first..=last {
                        out.push_str(self.line(l));
                    }
                }
            }
            next_line = last + 1;
        }

        for l in next_line..self.line_starts.len() {
            let line = self.line(l);
            if std::mem::take(&mut skip_sep) && line.trim() == "#" {
                continue;
            }
            out.push_str(line);
        }

        out
    }

    /// Write the document out from its parsed contents alone. For layouts we can't map back to
    /// source lines; comments are lost.
    fn to_text_rebuilt(&self) -> String {
        let mut out = String::with_capacity(self.text.len() + 4_096);

        for item in &self.items {
            match item {
                Item::Other { text, .. } => {
                    out.push_str(text);
                    out.push_str(self.newline);
                }
                Item::Category(cat) => {
                    if cat.is_empty() {
                        continue;
                    }
                    if let Some(sep) = &self.separator {
                        out.push_str(sep);
                        out.push_str(self.newline);
                    }
                    out.push_str(&cat.to_text(self.newline));
                }
            }
        }
        if let Some(sep) = &self.separator {
            out.push_str(sep);
            out.push_str(self.newline);
        }

        out
    }
}
