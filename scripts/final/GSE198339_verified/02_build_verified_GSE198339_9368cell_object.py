from pathlib import Path
import re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
DATA = PROJECT / "data/extracted"
OUT = PROJECT / "results/corrected_object"
OUT.mkdir(parents=True, exist_ok=True)

meta_files = sorted(DATA.glob("GSM*_Participant_*_metadata.csv.gz"))
expr_files = sorted(DATA.glob("GSM*_Participant_*_processed_gene_expression_data.csv.gz"))

assert len(meta_files) == 8, f"Expected 8 metadata files, found {len(meta_files)}"
assert len(expr_files) == 8, f"Expected 8 expression files, found {len(expr_files)}"

adatas = []
audit = []

for mf in meta_files:
    match = re.search(r"(GSM\d+)_(Participant_\d+)_metadata", mf.name)
    assert match, f"Cannot parse filename: {mf.name}"
    gsm, participant = match.group(1), match.group(2)

    ef = DATA / f"{gsm}_{participant}_processed_gene_expression_data.csv.gz"
    assert ef.exists(), f"Missing expression file: {ef}"

    meta = pd.read_csv(mf, index_col=0)
    expr = pd.read_csv(ef, index_col=0)

    assert expr.shape[1] == meta.shape[0], (
        f"{participant}: expression has {expr.shape[1]} cell columns; "
        f"metadata has {meta.shape[0]} rows."
    )

    expr = expr.T
    expr.index = expr.index.astype(str)
    meta.index = meta.index.astype(str)

    assert set(expr.index) == set(meta.index), f"{participant}: barcode mismatch"
    meta = meta.loc[expr.index].copy()

    meta["ParticipantID"] = participant
    meta["GSM"] = gsm
    meta["non_HDL"] = (
        pd.to_numeric(meta["cholesterol"], errors="raise")
        - pd.to_numeric(meta["HDL"], errors="raise")
    )

    prefixed_index = [f"{participant}_{bc}" for bc in meta.index]
    meta.index = prefixed_index
    expr.index = prefixed_index

    adata = ad.AnnData(
        X=sparse.csr_matrix(expr.to_numpy(dtype=np.float32)),
        obs=meta,
        var=pd.DataFrame(index=expr.columns.astype(str)),
    )
    adata.var_names_make_unique()
    adatas.append(adata)

    audit.append({
        "ParticipantID": participant,
        "AS_status": meta["Atherosclerosis Status"].iloc[0],
        "cholesterol": meta["cholesterol"].iloc[0],
        "HDL": meta["HDL"].iloc[0],
        "non_HDL": meta["non_HDL"].iloc[0],
        "total_cells": adata.n_obs,
        "NK_resting": int((meta["ClusterName"] == "NK cells resting").sum()),
        "NK_activated": int((meta["ClusterName"] == "NK cells activated").sum()),
    })

combined = ad.concat(adatas, join="outer", merge="same")
combined.var_names_make_unique()

audit_df = pd.DataFrame(audit).sort_values("ParticipantID")
audit_df["NK_total"] = audit_df["NK_resting"] + audit_df["NK_activated"]
audit_df["NK_proportion"] = audit_df["NK_total"] / audit_df["total_cells"]

assert combined.n_obs == 9368, f"Expected 9,368 cells, obtained {combined.n_obs}"
assert combined.obs["ParticipantID"].nunique() == 8
assert combined.obs_names.is_unique

raw_out = OUT / "GSE198339_official_processed_9368cells_raw.h5ad"
combined.write_h5ad(raw_out)

normalized = combined.copy()
sc.pp.normalize_total(normalized, target_sum=1e4)
sc.pp.log1p(normalized)
normalized.raw = normalized

sc.pp.highly_variable_genes(
    normalized, n_top_genes=min(3000, normalized.n_vars), flavor="seurat"
)
sc.tl.pca(normalized, use_highly_variable=True, svd_solver="arpack")
sc.pp.neighbors(normalized, n_neighbors=15, n_pcs=30)
sc.tl.umap(normalized)

norm_out = OUT / "GSE198339_official_processed_9368cells_normalized_umap.h5ad"
normalized.write_h5ad(norm_out)

audit_df.to_csv(OUT / "GSE198339_verified_participant_NK_summary.csv", index=False)

with open(OUT / "GSE198339_verified_object_audit.txt", "w") as handle:
    handle.write("GSE198339 verified object audit\n")
    handle.write("=" * 42 + "\n")
    handle.write("Source: official GEO processed expression and metadata CSV files\n")
    handle.write(f"Total cells: {combined.n_obs:,}\n")
    handle.write(f"Total genes: {combined.n_vars:,}\n")
    handle.write(f"Participants: {combined.obs['ParticipantID'].nunique()}\n")
    handle.write("Official ClusterName annotations retained.\n")
    handle.write("non-HDL calculated as cholesterol - HDL.\n")
    handle.write("No additional cell filtering applied; official processed retained-cell set used.\n\n")
    handle.write(audit_df.to_string(index=False))
    handle.write("\n")

print(audit_df.to_string(index=False))
print()
print("PASS: verified 9,368-cell object built from official processed data.")
print("Saved:", raw_out)
print("Saved:", norm_out)
print("Saved:", OUT / "GSE198339_verified_object_audit.txt")
