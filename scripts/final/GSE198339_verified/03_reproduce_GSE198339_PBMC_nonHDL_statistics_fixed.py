from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
INPUT = PROJECT / "results/corrected_object/GSE198339_official_processed_9368cells_normalized_umap.h5ad"
OUT = PROJECT / "results/corrected_pbmc_statistics"
OUT.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(INPUT)

required_obs = [
    "ParticipantID", "ClusterName", "non_HDL",
    "cholesterol", "HDL", "Atherosclerosis Status"
]
missing_obs = [c for c in required_obs if c not in adata.obs.columns]
if missing_obs:
    raise RuntimeError(f"Missing obs columns: {missing_obs}")

genes_needed = [
    "GZMK", "NKG7", "GNLY", "KLRD1",
    "TYROBP", "FCGR3A", "PRF1", "GZMB"
]
missing_genes = [g for g in genes_needed if g not in adata.var_names]
if missing_genes:
    raise RuntimeError(f"Missing expected genes: {missing_genes}")

nk_labels = ["NK cells resting", "NK cells activated"]
nk = adata[adata.obs["ClusterName"].isin(nk_labels)].copy()

if nk.n_obs == 0:
    raise RuntimeError("No NK cells found using official ClusterName annotations.")

def gene_vector(a, gene):
    values = a[:, gene].X
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()

for gene in genes_needed:
    nk.obs[f"{gene}_expr"] = gene_vector(nk, gene)

gzmk_like_genes = ["GZMK", "NKG7", "GNLY", "KLRD1"]
nk.obs["GZMK_like_score"] = nk.obs[
    [f"{g}_expr" for g in gzmk_like_genes]
].mean(axis=1)

cytotoxic_core_genes = ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"]
nk.obs["Cytotoxic_core_score"] = nk.obs[
    [f"{g}_expr" for g in cytotoxic_core_genes]
].mean(axis=1)

# Clinical data: preserve categorical status field; do not fill it with numeric values.
clinical = (
    adata.obs.groupby("ParticipantID", observed=True)
    .first()[["Atherosclerosis Status", "cholesterol", "HDL", "non_HDL"]]
    .copy()
)

total_cells = (
    adata.obs.groupby("ParticipantID", observed=True)
    .size()
    .rename("Total_cells")
)

nk_cells = (
    nk.obs.groupby("ParticipantID", observed=True)
    .size()
    .rename("NK_total_cells")
)

nk_resting_cells = (
    adata.obs.loc[adata.obs["ClusterName"] == "NK cells resting"]
    .groupby("ParticipantID", observed=True)
    .size()
    .rename("NK_resting_cells")
)

summary = clinical.join(total_cells).join(nk_cells).join(nk_resting_cells)

# Fill only numeric cell-count columns.
for col in ["NK_total_cells", "NK_resting_cells"]:
    summary[col] = summary[col].fillna(0).astype(int)

summary["Total_cells"] = summary["Total_cells"].astype(int)
summary["NK_total_proportion"] = summary["NK_total_cells"] / summary["Total_cells"]
summary["NK_resting_proportion"] = summary["NK_resting_cells"] / summary["Total_cells"]

metric_columns = (
    [f"{g}_expr" for g in genes_needed]
    + ["GZMK_like_score", "Cytotoxic_core_score"]
)

means = nk.obs.groupby("ParticipantID", observed=True)[metric_columns].mean()
summary = summary.join(means).reset_index()
summary = summary.sort_values("ParticipantID")

summary.to_csv(
    OUT / "GSE198339_verified_participant_level_NK_metrics.csv",
    index=False
)

tests = [
    ("NKG7 expression in NK cells", "NKG7_expr"),
    ("GZMK expression in NK cells", "GZMK_expr"),
    ("GNLY expression in NK cells", "GNLY_expr"),
    ("KLRD1 expression in NK cells", "KLRD1_expr"),
    ("TYROBP expression in NK cells", "TYROBP_expr"),
    ("GZMK-like composite score", "GZMK_like_score"),
    ("Cytotoxic core score", "Cytotoxic_core_score"),
    ("NK total-cell proportion", "NK_total_proportion"),
    ("NK resting-cell proportion", "NK_resting_proportion"),
]

results = []
for label, col in tests:
    values = summary[["non_HDL", col]].dropna()
    rho, p = spearmanr(values["non_HDL"], values[col])
    results.append({
        "Feature": label,
        "Metric_column": col,
        "n_participants": len(values),
        "Spearman_rho": rho,
        "p_value": p,
    })

stats = pd.DataFrame(results).sort_values("p_value")
stats.to_csv(
    OUT / "GSE198339_verified_nonHDL_NK_statistics.csv",
    index=False
)

print("VERIFIED PARTICIPANT-LEVEL NK METRICS")
print("=" * 100)
print(summary.to_string(index=False))
print()
print("VERIFIED NON-HDL CORRELATION STATISTICS")
print("=" * 100)
print(stats.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

with open(OUT / "GSE198339_verified_statistics_audit.txt", "w") as handle:
    handle.write("GSE198339 verified PBMC disease-context statistics audit\n")
    handle.write("=" * 62 + "\n")
    handle.write("Input object: official processed 9,368-cell normalized object\n")
    handle.write("Participants: 8\n")
    handle.write("NK labels: NK cells resting; NK cells activated\n")
    handle.write("non-HDL: cholesterol - HDL\n")
    handle.write("GZMK-like score: mean normalized expression of GZMK, NKG7, GNLY, KLRD1 within NK cells\n")
    handle.write("Categorical clinical metadata preserved; missing-value fill restricted to numeric count columns.\n\n")
    handle.write("Participant-level metrics:\n")
    handle.write(summary.to_string(index=False))
    handle.write("\n\nCorrelation statistics:\n")
    handle.write(stats.to_string(index=False))
    handle.write("\n")

print()
print("PASS: PBMC statistics reproduced from the verified 9,368-cell object.")
print("Saved outputs:")
print(OUT / "GSE198339_verified_participant_level_NK_metrics.csv")
print(OUT / "GSE198339_verified_nonHDL_NK_statistics.csv")
print(OUT / "GSE198339_verified_statistics_audit.txt")
