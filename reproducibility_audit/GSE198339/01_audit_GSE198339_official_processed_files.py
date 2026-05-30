from pathlib import Path
import csv
import gzip
import re
import sys

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
DATA = PROJECT / "data" / "extracted"
OUT = PROJECT / "results"
OUT.mkdir(parents=True, exist_ok=True)

metadata_files = sorted(DATA.glob("*_metadata.csv.gz"))
expr_files = sorted(DATA.glob("*_processed_gene_expression_data.csv.gz"))
h5_files = sorted(DATA.glob("*_raw_gene_bc_matrices_h5.h5"))

print("GSE198339 OFFICIAL PROCESSED-FILE AUDIT")
print("=" * 72)
print(f"Metadata files found:             {len(metadata_files)}")
print(f"Processed expression files found: {len(expr_files)}")
print(f"H5 files found:                   {len(h5_files)}")
print()

if len(metadata_files) != 8:
    raise RuntimeError(f"Expected 8 metadata files, found {len(metadata_files)}")
if len(expr_files) != 8:
    raise RuntimeError(f"Expected 8 processed expression files, found {len(expr_files)}")
if len(h5_files) != 8:
    raise RuntimeError(f"Expected 8 H5 files, found {len(h5_files)}")

def participant_id(filename: str) -> str:
    m = re.search(r"Participant_(\d+)", filename)
    return f"Participant_{m.group(1)}" if m else filename

def count_csv_rows_gz(path: Path) -> tuple[int, list[str]]:
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    return n_rows, header

def inspect_expression_orientation(path: Path) -> tuple[int, int, str]:
    """
    Returns data-row count, number of columns, and inferred orientation note.
    Cell count is inferred later by comparing dimensions with metadata count.
    """
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n_cols = len(header)
        n_rows = sum(1 for _ in reader)
    return n_rows, n_cols, header[0] if header else ""

rows = []

for meta in metadata_files:
    pid = participant_id(meta.name)
    paired_expr = next((f for f in expr_files if pid in f.name), None)
    if paired_expr is None:
        raise RuntimeError(f"No expression file matched {pid}")

    meta_n_rows, meta_header = count_csv_rows_gz(meta)
    expr_n_rows, expr_n_cols, expr_first_header = inspect_expression_orientation(paired_expr)

    # An expression CSV may have cells as columns plus one feature/index column,
    # or cells as rows. Match against metadata count explicitly.
    if expr_n_cols == meta_n_rows + 1:
        orientation = "genes x cells; first/index column plus matching cell columns"
        expr_cell_count = expr_n_cols - 1
        match = "PASS"
    elif expr_n_cols == meta_n_rows:
        orientation = "genes x cells; matching cell columns"
        expr_cell_count = expr_n_cols
        match = "PASS"
    elif expr_n_rows == meta_n_rows:
        orientation = "cells x genes; matching cell rows"
        expr_cell_count = expr_n_rows
        match = "PASS"
    else:
        orientation = "orientation/count not matched automatically"
        expr_cell_count = ""
        match = "CHECK"

    rows.append([
        pid,
        meta.name,
        meta_n_rows,
        paired_expr.name,
        expr_n_rows,
        expr_n_cols,
        expr_cell_count,
        match,
        orientation
    ])

total_meta_cells = sum(int(r[2]) for r in rows)
matched_expr_cells = sum(int(r[6]) for r in rows if r[6] != "")

summary_file = OUT / "GSE198339_official_processed_cell_count_summary.tsv"
with summary_file.open("w", newline="") as fh:
    writer = csv.writer(fh, delimiter="\t")
    writer.writerow([
        "participant", "metadata_file", "metadata_cell_rows",
        "expression_file", "expression_data_rows", "expression_columns",
        "inferred_expression_cells", "metadata_expression_match", "orientation"
    ])
    writer.writerows(rows)

for r in rows:
    print(f"{r[0]}: metadata cells={r[2]:,}; "
          f"expression dimensions={r[4]:,} rows x {r[5]:,} columns; "
          f"match={r[7]}")
    print(f"  {r[8]}")

print()
print("=" * 72)
print(f"Official GEO-stated PBMC count:                   9,368")
print(f"Total cells counted from downloaded metadata:    {total_meta_cells:,}")
if all(r[7] == "PASS" for r in rows):
    print(f"Total matching processed-expression cell count:  {matched_expr_cells:,}")
else:
    print("Processed-expression count requires manual orientation review.")
print()

if total_meta_cells == 9368:
    print("PASS: The downloaded official metadata files sum to the GEO-stated total of 9,368 cells.")
else:
    print("ALERT: The downloaded metadata total does not match GEO's stated 9,368 cells.")

if all(r[7] == "PASS" for r in rows) and matched_expr_cells == total_meta_cells:
    print("PASS: Processed expression-file dimensions match metadata cell counts.")
else:
    print("CHECK: At least one processed expression file does not automatically match metadata.")

audit_txt = OUT / "GSE198339_official_cell_count_audit.txt"
with audit_txt.open("w") as fh:
    fh.write("GSE198339 official processed-file cell-count audit\n")
    fh.write("=" * 55 + "\n")
    fh.write("GEO-reported source-dataset PBMC count: 9,368\n")
    fh.write(f"Metadata rows counted from downloaded files: {total_meta_cells:,}\n")
    if all(r[7] == "PASS" for r in rows):
        fh.write(f"Expression cells matched to metadata: {matched_expr_cells:,}\n")
    fh.write(f"All metadata-expression pairs matched: {all(r[7] == 'PASS' for r in rows)}\n")
    fh.write("\nParticipant counts:\n")
    for r in rows:
        fh.write(f"{r[0]}\t{r[2]}\t{r[7]}\t{r[8]}\n")

print()
print(f"Saved: {summary_file}")
print(f"Saved: {audit_txt}")
