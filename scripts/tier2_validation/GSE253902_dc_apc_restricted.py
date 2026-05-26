from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE253902_merged_gex_raw.h5ad").copy()

# Normalize/log if not already done in object loaded here
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Marker modules
cdc2_genes = ["HLA-DRA", "HLA-DRB1", "CD74", "FCER1A", "CD1C", "CLEC10A"]
pdc_genes = ["GZMB", "TCF4", "JCHAIN", "IRF7"]
myeloid_genes = ["LST1", "CTSS", "FCN1", "TYMP", "S100A8", "S100A9"]
apc_seed_genes = ["HLA-DRA", "CD74", "FCER1A", "CD1C", "CLEC10A", "LST1", "CTSS"]

present_cdc2 = [g for g in cdc2_genes if g in adata.var_names]
present_pdc = [g for g in pdc_genes if g in adata.var_names]
present_myeloid = [g for g in myeloid_genes if g in adata.var_names]
present_apc = [g for g in apc_seed_genes if g in adata.var_names]

sc.tl.score_genes(adata, present_cdc2, score_name="cdc2_score", use_raw=False)
sc.tl.score_genes(adata, present_pdc, score_name="pdc_score", use_raw=False)
sc.tl.score_genes(adata, present_myeloid, score_name="myeloid_score", use_raw=False)
sc.tl.score_genes(adata, present_apc, score_name="apc_seed_score", use_raw=False)

obs = adata.obs[["sample", "gsm", "status", "cdc2_score", "pdc_score", "myeloid_score", "apc_seed_score"]].copy()
obs["cdc2_minus_inflammatory"] = obs["cdc2_score"] - obs["myeloid_score"]
obs["pdc_minus_cdc2"] = obs["pdc_score"] - obs["cdc2_score"]

# Define APC-enriched cells globally
apc_thresh = obs["apc_seed_score"].quantile(0.80)
obs["is_apc_enriched"] = obs["apc_seed_score"] >= apc_thresh

apc = obs[obs["is_apc_enriched"]].copy()

# Within APC-enriched cells define high states globally
cdc2_high_thresh = apc["cdc2_score"].quantile(0.75)
myeloid_high_thresh = apc["myeloid_score"].quantile(0.75)
pdc_high_thresh = apc["pdc_score"].quantile(0.75)

apc["is_cdc2_high"] = apc["cdc2_score"] >= cdc2_high_thresh
apc["is_myeloid_high"] = apc["myeloid_score"] >= myeloid_high_thresh
apc["is_pdc_high"] = apc["pdc_score"] >= pdc_high_thresh

obs.to_csv(outdir / "GSE253902_dc_apc_all_cells.csv")
apc.to_csv(outdir / "GSE253902_dc_apc_subset.csv")

# Per-sample summaries
all_sample = (
    obs.groupby(["sample", "gsm", "status"], observed=True)
    .agg(
        n_cells=("sample", "size"),
        prop_apc=("is_apc_enriched", "mean"),
    )
    .reset_index()
)

apc_sample = (
    apc.groupby(["sample", "gsm", "status"], observed=True)
    .agg(
        n_apc_cells=("sample", "size"),
        cdc2_score_mean=("cdc2_score", "mean"),
        pdc_score_mean=("pdc_score", "mean"),
        myeloid_score_mean=("myeloid_score", "mean"),
        cdc2_minus_inflammatory_mean=("cdc2_minus_inflammatory", "mean"),
        pdc_minus_cdc2_mean=("pdc_minus_cdc2", "mean"),
        prop_cdc2_high=("is_cdc2_high", "mean"),
        prop_myeloid_high=("is_myeloid_high", "mean"),
        prop_pdc_high=("is_pdc_high", "mean"),
    )
    .reset_index()
)

sample_df = all_sample.merge(apc_sample, on=["sample", "gsm", "status"], how="left")
sample_df.to_csv(outdir / "GSE253902_dc_apc_restricted_sample_summary.csv", index=False)

# Compare the single asymptomatic sample to symptomatic sample distribution
asym = sample_df[sample_df["status"] == "Asymptomatic"].copy()
sym = sample_df[sample_df["status"] == "Symptomatic"].copy()

outcomes = [
    "prop_apc",
    "cdc2_score_mean",
    "myeloid_score_mean",
    "cdc2_minus_inflammatory_mean",
    "prop_cdc2_high",
    "prop_myeloid_high",
]

rows = []
for outcome in outcomes:
    a = pd.to_numeric(asym[outcome], errors="coerce").iloc[0]
    s = pd.to_numeric(sym[outcome], errors="coerce").dropna()

    sym_mean = s.mean()
    sym_sd = s.std(ddof=1) if len(s) > 1 else np.nan
    z = (a - sym_mean) / sym_sd if pd.notna(sym_sd) and sym_sd != 0 else np.nan
    percentile = (s < a).mean() if len(s) else np.nan

    rows.append({
        "outcome": outcome,
        "asymptomatic_value": a,
        "symptomatic_mean": sym_mean,
        "symptomatic_sd": sym_sd,
        "asym_minus_sym_mean": a - sym_mean,
        "z_vs_symptomatic_distribution": z,
        "percentile_vs_symptomatic_distribution": percentile,
    })

compare_df = pd.DataFrame(rows)
compare_df.to_csv(outdir / "GSE253902_dc_apc_restricted_asym_vs_sym_reference.csv", index=False)

print("APC threshold:", apc_thresh)
print("cDC2-high threshold:", cdc2_high_thresh)
print("Myeloid-high threshold:", myeloid_high_thresh)
print("pDC-high threshold:", pdc_high_thresh)

print("\nSample summary:")
print(sample_df.to_string(index=False))

print("\nAsymptomatic vs symptomatic reference comparison:")
print(compare_df.to_string(index=False))

print("\nSaved:")
print(outdir / "GSE253902_dc_apc_restricted_sample_summary.csv")
print(outdir / "GSE253902_dc_apc_restricted_asym_vs_sym_reference.csv")
