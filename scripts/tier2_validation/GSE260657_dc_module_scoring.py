from pathlib import Path
import pandas as pd
import numpy as np

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE260657/extracted")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

# Sample symptom mapping from GEO metadata
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

# Marker modules
cdc2_genes = ["HLA-DRA", "HLA-DRB1", "CD74", "FCER1A", "CD1C", "CLEC10A"]
pdc_genes = ["GZMB", "TCF4", "JCHAIN", "IRF7"]
myeloid_genes = ["LST1", "CTSS", "FCN1", "S100A8", "S100A9", "TYMP"]

cell_rows = []
sample_rows = []

files = sorted(base.glob("GSM*_athero_human_*.txt.gz"))

for f in files:
    sample_name = f.name.replace(".txt.gz", "").split("_", 1)[1]
    status = sample_status[sample_name]

    print(f"Reading {f.name}")
    df = pd.read_csv(f, sep="\t", index_col=0)

    # Keep only genes present
    cdc2_present = [g for g in cdc2_genes if g in df.index]
    pdc_present = [g for g in pdc_genes if g in df.index]
    myeloid_present = [g for g in myeloid_genes if g in df.index]

    # Cell-level module scores = mean normalized-by-library-count expression
    libsize = df.sum(axis=0)
    norm = df.divide(libsize, axis=1) * 1e4
    norm = np.log1p(norm)

    cdc2_score = norm.loc[cdc2_present].mean(axis=0) if len(cdc2_present) else pd.Series(np.nan, index=norm.columns)
    pdc_score = norm.loc[pdc_present].mean(axis=0) if len(pdc_present) else pd.Series(np.nan, index=norm.columns)
    myeloid_score = norm.loc[myeloid_present].mean(axis=0) if len(myeloid_present) else pd.Series(np.nan, index=norm.columns)

    # Cell-level output
    cell_df = pd.DataFrame({
        "cell_barcode": norm.columns.astype(str),
        "sample": sample_name,
        "status": status,
        "cdc2_score": cdc2_score.values,
        "pdc_score": pdc_score.values,
        "myeloid_score": myeloid_score.values,
        "pdc_minus_cdc2": pdc_score.values - cdc2_score.values,
    })
    cell_rows.append(cell_df)

    # Sample-level summaries
    sample_rows.append({
        "sample": sample_name,
        "status": status,
        "n_cells": norm.shape[1],
        "cdc2_score_mean": float(cdc2_score.mean()),
        "pdc_score_mean": float(pdc_score.mean()),
        "myeloid_score_mean": float(myeloid_score.mean()),
        "pdc_minus_cdc2_mean": float((pdc_score - cdc2_score).mean()),
        "prop_top_quartile_cdc2": float((cdc2_score >= cdc2_score.quantile(0.75)).mean()),
        "prop_top_quartile_pdc": float((pdc_score >= pdc_score.quantile(0.75)).mean()),
    })

cell_out = pd.concat(cell_rows, ignore_index=True)
sample_out = pd.DataFrame(sample_rows)

cell_out.to_csv(outdir / "GSE260657_dc_module_scores_cell_level.csv", index=False)
sample_out.to_csv(outdir / "GSE260657_dc_module_scores_sample_level.csv", index=False)

print("\nSample-level summary:")
print(sample_out.to_string(index=False))
print("\nSaved:")
print(outdir / "GSE260657_dc_module_scores_cell_level.csv")
print(outdir / "GSE260657_dc_module_scores_sample_level.csv")
