import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
raw_file = project_dir / "data" / "first_dataset" / "raw" / "human_immune_health_atlas_mono.h5ad"
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"

results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading: {raw_file}")
adata = sc.read_h5ad(raw_file)

# Use raw counts/expression if present
if adata.raw is not None:
    print("Using adata.raw.to_adata() for module scoring")
    adata = adata.raw.to_adata()
else:
    print("adata.raw not found; using adata directly")

# Keep key metadata from original obs if needed
# If adata.raw.to_adata() preserved obs, this is fine
required_cols = ["subject.subjectGuid", "subject.ageGroup", "AIFI_L2", "AIFI_L3"]
missing = [c for c in required_cols if c not in adata.obs.columns]
if missing:
    raise ValueError(f"Missing required obs columns: {missing}")

# Basic normalization for scoring
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Gene symbols available
genes_present = set(map(str, adata.var_names))

modules = {
    "heatshock_score": [
        "HSPA1A","HSPA1B","HSPH1","DNAJB1","HSP90AA1","HSPB1","BAG3","HSPD1"
    ],
    "oxidative_stress_score": [
        "NQO1","HMOX1","TXNRD1","GCLM","GCLC","SOD2","PRDX1","SRXN1"
    ],
    "interferon_score": [
        "ISG15","IFI6","IFIT1","IFIT2","IFIT3","MX1","MX2","OAS1","OAS3","IFI44L"
    ],
    "nfkb_inflammatory_score": [
        "IL1B","TNF","NFKBIA","CXCL8","CCL3","CCL4","PTGS2","TNFAIP3"
    ],
    "proteostasis_score": [
        "HSPA5","XBP1","DDIT3","ATF4","DNAJB9","HERPUD1","EIF2AK3","CALR"
    ],
}

module_summary = []
used_modules = {}

for score_name, gene_list in modules.items():
    present = [g for g in gene_list if g in genes_present]
    used_modules[score_name] = present
    print(f"{score_name}: {len(present)}/{len(gene_list)} genes present")
    if len(present) >= 3:
        sc.tl.score_genes(adata, gene_list=present, score_name=score_name, use_raw=False)
        module_summary.append({
            "module": score_name,
            "n_genes_requested": len(gene_list),
            "n_genes_present": len(present),
            "genes_used": ",".join(present)
        })
    else:
        print(f"Skipping {score_name} because too few genes are present")

module_summary_df = pd.DataFrame(module_summary)
module_summary_df.to_csv(results_dir / "monocyte_module_gene_coverage.csv", index=False)

score_cols = [m for m in modules if m in adata.obs.columns]

obs_cols = ["subject.subjectGuid", "subject.ageGroup", "AIFI_L2", "AIFI_L3"] + score_cols
obs = adata.obs[obs_cols].copy()

# Keep only Young/Older for main comparison
obs_main = obs[obs["subject.ageGroup"].isin(["Young Adult", "Older Adult"])].copy()
obs_main.to_csv(results_dir / "monocyte_module_scores_per_cell.csv", index=True)

# Donor-level mean scores across all monocytes
donor_scores = (
    obs_main.groupby(["subject.subjectGuid", "subject.ageGroup"], observed=True)[score_cols]
    .mean()
    .reset_index()
)
donor_scores.to_csv(results_dir / "monocyte_module_scores_per_donor.csv", index=False)

# Donor-level mean scores within AIFI_L2
donor_l2_scores = (
    obs_main.groupby(["subject.subjectGuid", "subject.ageGroup", "AIFI_L2"], observed=True)[score_cols]
    .mean()
    .reset_index()
)
donor_l2_scores.to_csv(results_dir / "monocyte_module_scores_per_donor_AIFI_L2.csv", index=False)

# Donor-level mean scores within AIFI_L3
donor_l3_scores = (
    obs_main.groupby(["subject.subjectGuid", "subject.ageGroup", "AIFI_L3"], observed=True)[score_cols]
    .mean()
    .reset_index()
)
donor_l3_scores.to_csv(results_dir / "monocyte_module_scores_per_donor_AIFI_L3.csv", index=False)

# Simple donor-level stats across all monocytes
rows = []
for score in score_cols:
    y = donor_scores.loc[donor_scores["subject.ageGroup"] == "Young Adult", score].dropna()
    o = donor_scores.loc[donor_scores["subject.ageGroup"] == "Older Adult", score].dropna()

    pval = mannwhitneyu(y, o, alternative="two-sided").pvalue if len(y) and len(o) else np.nan
    rows.append({
        "module": score,
        "young_n": len(y),
        "older_n": len(o),
        "young_mean": y.mean() if len(y) else np.nan,
        "older_mean": o.mean() if len(o) else np.nan,
        "delta_older_minus_young": (o.mean() - y.mean()) if len(y) and len(o) else np.nan,
        "p_value": pval
    })

stats = pd.DataFrame(rows).sort_values("delta_older_minus_young", ascending=False)
stats.to_csv(results_dir / "monocyte_module_scores_stats.csv", index=False)

print("Done.")
print(results_dir / "monocyte_module_gene_coverage.csv")
print(results_dir / "monocyte_module_scores_per_donor.csv")
print(results_dir / "monocyte_module_scores_stats.csv")
