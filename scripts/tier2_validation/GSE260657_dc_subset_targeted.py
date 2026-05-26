from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE260657/extracted")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

sample_status = {
    "athero_human_1": "Asymptomatic",
    "athero_human_2": "Asymptomatic",
    "athero_human_3": "Asymptomatic",
    "athero_human_4": "Symptomatic",
    "athero_human_5": "Asymptomatic",
    "athero_human_6": "Symptomatic",
    "athero_human_7": "Symptomatic",
    "athero_human_8": "Symptomatic",
    "athero_human_9": "Asymptomatic",
    "athero_human_10": "Symptomatic",
    "athero_human_11": "Symptomatic",
    "athero_human_12": "Symptomatic",
    "athero_human_13": "Asymptomatic",
    "athero_human_14": "Asymptomatic",
    "athero_human_15": "Symptomatic",
}

# marker modules
cdc2_genes = ["HLA-DRA", "HLA-DRB1", "CD74", "FCER1A", "CD1C", "CLEC10A"]
pdc_genes = ["GZMB", "TCF4", "JCHAIN", "IRF7"]
myeloid_genes = ["LST1", "CTSS", "FCN1", "S100A8", "S100A9", "TYMP"]

# broad APC / DC-enrichment markers
apc_seed_genes = ["HLA-DRA", "CD74", "FCER1A", "CD1C", "CLEC10A", "LST1", "CTSS"]

cell_rows = []

files = sorted(base.glob("GSM*_athero_human_*.txt.gz"))
for f in files:
    sample_name = f.name.replace(".txt.gz", "").split("_", 1)[1]
    status = sample_status[sample_name]

    print(f"Reading {f.name}")
    df = pd.read_csv(f, sep="\t", index_col=0)

    # library-size normalize then log1p
    libsize = df.sum(axis=0)
    norm = df.divide(libsize, axis=1) * 1e4
    norm = np.log1p(norm)

    def module_mean(glist):
        present = [g for g in glist if g in norm.index]
        if not present:
            return pd.Series(np.nan, index=norm.columns)
        return norm.loc[present].mean(axis=0)

    cdc2_score = module_mean(cdc2_genes)
    pdc_score = module_mean(pdc_genes)
    myeloid_score = module_mean(myeloid_genes)
    apc_seed_score = module_mean(apc_seed_genes)

    cell_df = pd.DataFrame({
        "cell_barcode": norm.columns.astype(str),
        "sample": sample_name,
        "status": status,
        "cdc2_score": cdc2_score.values,
        "pdc_score": pdc_score.values,
        "myeloid_score": myeloid_score.values,
        "apc_seed_score": apc_seed_score.values,
    })
    cell_df["pdc_minus_cdc2"] = cell_df["pdc_score"] - cell_df["cdc2_score"]
    cell_rows.append(cell_df)

cells = pd.concat(cell_rows, ignore_index=True)

# Define likely APC/myeloid/DC-enriched subset globally
apc_thresh = cells["apc_seed_score"].quantile(0.80)
cells["is_apc_enriched"] = cells["apc_seed_score"] >= apc_thresh

# Within APC-enriched cells, define cDC2-like and pDC-like high states
apc_cells = cells[cells["is_apc_enriched"]].copy()

cdc2_thresh = apc_cells["cdc2_score"].quantile(0.75)
pdc_thresh = apc_cells["pdc_score"].quantile(0.75)

apc_cells["is_cdc2_high"] = apc_cells["cdc2_score"] >= cdc2_thresh
apc_cells["is_pdc_high"] = apc_cells["pdc_score"] >= pdc_thresh

cells.to_csv(outdir / "GSE260657_dc_targeted_cell_scores_all.csv", index=False)
apc_cells.to_csv(outdir / "GSE260657_dc_targeted_apc_subset.csv", index=False)

# Sample-level summaries within APC-enriched subset
sample_rows = []
for sample, sub in apc_cells.groupby("sample"):
    status = sub["status"].iloc[0]
    sample_rows.append({
        "sample": sample,
        "status": status,
        "n_apc_cells": len(sub),
        "cdc2_score_mean": sub["cdc2_score"].mean(),
        "pdc_score_mean": sub["pdc_score"].mean(),
        "myeloid_score_mean": sub["myeloid_score"].mean(),
        "pdc_minus_cdc2_mean": sub["pdc_minus_cdc2"].mean(),
        "prop_cdc2_high": sub["is_cdc2_high"].mean(),
        "prop_pdc_high": sub["is_pdc_high"].mean(),
    })

sample_df = pd.DataFrame(sample_rows)
sample_df.to_csv(outdir / "GSE260657_dc_targeted_sample_summary.csv", index=False)

# Group stats
outcomes = [
    "cdc2_score_mean",
    "pdc_score_mean",
    "myeloid_score_mean",
    "pdc_minus_cdc2_mean",
    "prop_cdc2_high",
    "prop_pdc_high",
]

stats_rows = []
for outcome in outcomes:
    a = pd.to_numeric(sample_df.loc[sample_df["status"] == "Asymptomatic", outcome], errors="coerce").dropna()
    s = pd.to_numeric(sample_df.loc[sample_df["status"] == "Symptomatic", outcome], errors="coerce").dropna()
    p = mannwhitneyu(a, s, alternative="two-sided").pvalue if len(a) and len(s) else np.nan
    stats_rows.append({
        "outcome": outcome,
        "asymptomatic_mean": a.mean(),
        "symptomatic_mean": s.mean(),
        "symptomatic_minus_asymptomatic_mean": s.mean() - a.mean(),
        "mannwhitney_p": p,
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(outdir / "GSE260657_dc_targeted_group_stats.csv", index=False)

print("\nAPC-enriched threshold:", apc_thresh)
print("cDC2-high threshold:", cdc2_thresh)
print("pDC-high threshold:", pdc_thresh)

print("\nSample summary:")
print(sample_df.to_string(index=False))

print("\nGroup stats:")
print(stats_df.to_string(index=False))

print("\nSaved:")
print(outdir / "GSE260657_dc_targeted_apc_subset.csv")
print(outdir / "GSE260657_dc_targeted_sample_summary.csv")
print(outdir / "GSE260657_dc_targeted_group_stats.csv")
