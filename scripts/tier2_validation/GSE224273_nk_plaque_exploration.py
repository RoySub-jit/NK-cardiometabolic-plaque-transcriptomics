from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE224273_merged_rna_raw.h5ad").copy()

# normalize/log
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

nk_core_genes = ["NKG7", "GNLY", "PRF1", "GZMB", "KLRD1", "FCGR3A", "TYROBP"]
gzmk_like_genes = ["GZMK", "NKG7", "KLRD1", "GNLY"]
nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]

present_nk_core = [g for g in nk_core_genes if g in adata.var_names]
present_gzmk_like = [g for g in gzmk_like_genes if g in adata.var_names]
present_nk_seed = [g for g in nk_seed_genes if g in adata.var_names]

sc.tl.score_genes(adata, present_nk_core, score_name="nk_core_score", use_raw=False)
sc.tl.score_genes(adata, present_gzmk_like, score_name="gzmk_like_score", use_raw=False)
sc.tl.score_genes(adata, present_nk_seed, score_name="nk_seed_score", use_raw=False)

obs = adata.obs[["sample", "gsm", "status", "nk_core_score", "gzmk_like_score", "nk_seed_score"]].copy()

# define NK-high cells globally
nk_high_thresh = obs["nk_seed_score"].quantile(0.90)
obs["is_nk_high"] = obs["nk_seed_score"] >= nk_high_thresh

obs.to_csv(outdir / "GSE224273_nk_plaque_cell_scores.csv")

sample_df = (
    obs.groupby(["sample", "gsm", "status"], observed=True)
    .agg(
        n_cells=("sample", "size"),
        nk_core_score_mean=("nk_core_score", "mean"),
        gzmk_like_score_mean=("gzmk_like_score", "mean"),
        nk_seed_score_mean=("nk_seed_score", "mean"),
        prop_nk_high=("is_nk_high", "mean"),
    )
    .reset_index()
)

sample_df.to_csv(outdir / "GSE224273_nk_plaque_sample_scores.csv", index=False)

# group summary
rows = []
for endpoint in ["nk_core_score_mean", "gzmk_like_score_mean", "prop_nk_high"]:
    a = pd.to_numeric(sample_df.loc[sample_df["status"]=="Asymptomatic", endpoint], errors="coerce").dropna()
    s = pd.to_numeric(sample_df.loc[sample_df["status"]=="Symptomatic", endpoint], errors="coerce").dropna()
    rows.append({
        "endpoint": endpoint,
        "asymptomatic_mean": a.mean(),
        "symptomatic_mean": s.mean(),
        "symptomatic_minus_asymptomatic_mean": s.mean() - a.mean(),
        "asymptomatic_median": a.median(),
        "symptomatic_median": s.median(),
    })

group_df = pd.DataFrame(rows)
group_df.to_csv(outdir / "GSE224273_nk_plaque_group_summary.csv", index=False)

print("Present NK core genes:", present_nk_core)
print("Present GZMK-like genes:", present_gzmk_like)
print("Present NK seed genes:", present_nk_seed)
print("NK-high threshold:", nk_high_thresh)

print("\nSample summary:")
print(sample_df.to_string(index=False))

print("\nGroup summary:")
print(group_df.to_string(index=False))

print("\nSaved:")
print(outdir / "GSE224273_nk_plaque_sample_scores.csv")
print(outdir / "GSE224273_nk_plaque_group_summary.csv")
