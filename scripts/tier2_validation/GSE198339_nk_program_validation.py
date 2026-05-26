from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE198339/extracted")

# Genes to inspect
genes_of_interest = [
    "GZMK", "NKG7", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRD1", "TYROBP"
]

# Load combined metadata
meta = pd.read_csv(base / "GSE198339_all_metadata_combined.csv", index_col=0)

# Restrict to cleaner NK clusters
nk_meta = meta[meta["ClusterName"].isin(["NK cells resting", "NK cells activated"])].copy()

print("NK metadata shape:", nk_meta.shape)
print("\nNK cluster counts:")
print(nk_meta["ClusterName"].value_counts())

# The metadata index is the cell barcode
nk_barcodes = set(nk_meta.index.astype(str))

# Collect participant-level expression summaries
rows = []

expr_files = sorted(base.glob("*_processed_gene_expression_data.csv.gz"))
for f in expr_files:
    participant_guess = f.name.split("_processed_gene_expression_data")[0]
    print(f"\nReading {f.name}")

    # Load processed expression
    expr = pd.read_csv(f, index_col=0)

    # Standard orientation assumption: rows = genes, columns = cells
    # Keep only wanted genes present
    present_genes = [g for g in genes_of_interest if g in expr.index]
    if len(present_genes) == 0:
        print("No genes of interest found in", f.name)
        continue

    # Match columns to NK barcodes
    matched_cols = [c for c in expr.columns.astype(str) if c in nk_barcodes]
    if len(matched_cols) == 0:
        print("No NK barcodes matched in", f.name)
        continue

    sub_expr = expr.loc[present_genes, matched_cols].copy()

    # Metadata for matched cells
    sub_meta = nk_meta.loc[matched_cols].copy()

    # infer participant
    participant_ids = sub_meta["ParticipantID"].astype(str).unique().tolist()
    if len(participant_ids) != 1:
        print("Warning: multiple participant IDs found in matched cells:", participant_ids)

    participant_id = participant_ids[0]
    ath = sub_meta["Atherosclerosis Status"].iloc[0]
    chol = pd.to_numeric(sub_meta["cholesterol"].iloc[0], errors="coerce")
    hdl = pd.to_numeric(sub_meta["HDL"].iloc[0], errors="coerce")
    non_hdl = chol - hdl if pd.notna(chol) and pd.notna(hdl) else np.nan

    # Per-gene means
    gene_means = sub_expr.mean(axis=1).to_dict()

    # Composite means
    cytotoxic_core = [g for g in ["NKG7", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRD1", "TYROBP"] if g in present_genes]
    gzmk_like = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1"] if g in present_genes]

    row = {
        "ParticipantID": participant_id,
        "Atherosclerosis Status": ath,
        "cholesterol": chol,
        "HDL": hdl,
        "non_HDL": non_hdl,
        "n_nk_cells": len(matched_cols),
        "n_resting_nk": (sub_meta["ClusterName"] == "NK cells resting").sum(),
        "n_activated_nk": (sub_meta["ClusterName"] == "NK cells activated").sum(),
    }

    for g in genes_of_interest:
        row[g] = gene_means.get(g, np.nan)

    row["cytotoxic_core_score"] = np.nanmean([gene_means[g] for g in cytotoxic_core]) if len(cytotoxic_core) else np.nan
    row["gzmk_like_score"] = np.nanmean([gene_means[g] for g in gzmk_like]) if len(gzmk_like) else np.nan

    rows.append(row)

pt = pd.DataFrame(rows).sort_values("ParticipantID")
pt.to_csv(base / "GSE198339_nk_program_participant_level.csv", index=False)

print("\nParticipant-level NK expression summary:")
print(pt.to_string(index=False))

# Stats
outcomes = genes_of_interest + ["cytotoxic_core_score", "gzmk_like_score"]
stats_rows = []

for outcome in outcomes:
    vals = pd.to_numeric(pt[outcome], errors="coerce")
    non_hdl = pd.to_numeric(pt["non_HDL"], errors="coerce")

    mask = vals.notna() & non_hdl.notna()
    if mask.sum() >= 3:
        rho, p = spearmanr(non_hdl[mask], vals[mask])
    else:
        rho, p = np.nan, np.nan

    as_neg = vals[pt["Atherosclerosis Status"] == "AS-"].dropna()
    as_pos = vals[pt["Atherosclerosis Status"] == "AS+"].dropna()
    mw_p = mannwhitneyu(as_neg, as_pos, alternative="two-sided").pvalue if len(as_neg) and len(as_pos) else np.nan

    stats_rows.append({
        "outcome": outcome,
        "AS_minus_mean": as_neg.mean() if len(as_neg) else np.nan,
        "AS_plus_mean": as_pos.mean() if len(as_pos) else np.nan,
        "AS_plus_minus_AS_minus": (as_pos.mean() - as_neg.mean()) if len(as_neg) and len(as_pos) else np.nan,
        "mannwhitney_p": mw_p,
        "spearman_rho_non_HDL": rho,
        "spearman_p_non_HDL": p,
    })

stats = pd.DataFrame(stats_rows).sort_values("spearman_p_non_HDL", na_position="last")
stats.to_csv(base / "GSE198339_nk_program_validation_stats.csv", index=False)

print("\nNK program validation stats:")
print(stats.to_string(index=False))
print("\nSaved:")
print(base / "GSE198339_nk_program_participant_level.csv")
print(base / "GSE198339_nk_program_validation_stats.csv")
