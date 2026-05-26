from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from itertools import combinations

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE224273_merged_rna_raw.h5ad").copy()

# Normalize/log
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

# Define NK-high cells globally
nk_high_thresh = obs["nk_seed_score"].quantile(0.90)
obs["is_nk_high"] = obs["nk_seed_score"] >= nk_high_thresh
nk = obs[obs["is_nk_high"]].copy()

obs.to_csv(outdir / "GSE224273_nk_all_cells.csv")
nk.to_csv(outdir / "GSE224273_nk_high_cells.csv")

sample_df = (
    nk.groupby(["sample", "gsm", "status"], observed=True)
    .agg(
        n_nk_high_cells=("sample", "size"),
        nk_core_score_mean=("nk_core_score", "mean"),
        gzmk_like_score_mean=("gzmk_like_score", "mean"),
    )
    .reset_index()
)
sample_df.to_csv(outdir / "GSE224273_nk_high_sample_summary.csv", index=False)

def exact_permutation_p(values, labels):
    values = np.array(values, dtype=float)
    labels = np.array(labels)
    idx = np.arange(len(values))
    asym_idx = np.where(labels == "Asymptomatic")[0]
    sym_idx = np.where(labels == "Symptomatic")[0]
    obs_diff = values[asym_idx].mean() - values[sym_idx].mean()

    n_asym = len(asym_idx)
    diffs = []
    for comb in combinations(idx, n_asym):
        comb = np.array(comb)
        rest = np.setdiff1d(idx, comb)
        diffs.append(values[comb].mean() - values[rest].mean())
    diffs = np.array(diffs)
    p = np.mean(np.abs(diffs) >= abs(obs_diff))
    return obs_diff, p

rng = np.random.default_rng(42)

def bootstrap_ci(asym, sym, n_boot=5000):
    asym = np.array(asym, dtype=float)
    sym = np.array(sym, dtype=float)
    diffs = []
    for _ in range(n_boot):
        a = rng.choice(asym, size=len(asym), replace=True)
        s = rng.choice(sym, size=len(sym), replace=True)
        diffs.append(a.mean() - s.mean())
    diffs = np.array(diffs)
    return diffs.mean(), np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)

rows = []
for endpoint in ["gzmk_like_score_mean", "nk_core_score_mean"]:
    a = pd.to_numeric(sample_df.loc[sample_df["status"] == "Asymptomatic", endpoint], errors="coerce").dropna().values
    s = pd.to_numeric(sample_df.loc[sample_df["status"] == "Symptomatic", endpoint], errors="coerce").dropna().values

    if len(a) >= 1 and len(s) >= 1:
        diff, p = exact_permutation_p(np.r_[a, s], np.array(["Asymptomatic"] * len(a) + ["Symptomatic"] * len(s)))
        boot_mean, ci_low, ci_high = bootstrap_ci(a, s, n_boot=5000)
    else:
        diff, p, boot_mean, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan, np.nan

    rows.append({
        "endpoint": endpoint,
        "n_asymptomatic": len(a),
        "n_symptomatic": len(s),
        "asymptomatic_mean": a.mean() if len(a) else np.nan,
        "symptomatic_mean": s.mean() if len(s) else np.nan,
        "asym_minus_symptomatic_mean": diff,
        "exact_permutation_p": p,
        "bootstrap_mean_diff": boot_mean,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    })

res = pd.DataFrame(rows)
res.to_csv(outdir / "GSE224273_nk_high_targeted_stats.csv", index=False)

print("Present NK core genes:", present_nk_core)
print("Present GZMK-like genes:", present_gzmk_like)
print("Present NK seed genes:", present_nk_seed)
print("NK-high threshold:", nk_high_thresh)

print("\nNK-high sample summary:")
print(sample_df.to_string(index=False))

print("\nTargeted NK-high stats:")
print(res.to_string(index=False))

print("\nSaved:")
print(outdir / "GSE224273_nk_high_sample_summary.csv")
print(outdir / "GSE224273_nk_high_targeted_stats.csv")
