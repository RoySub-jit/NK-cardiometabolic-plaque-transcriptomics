import scanpy as sc
import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
raw_file = project_dir / "data" / "first_dataset" / "raw" / "human_immune_health_atlas_mono.h5ad"
results_dir = project_dir / "results" / "first_dataset"

results_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading: {raw_file}")
adata = sc.read_h5ad(raw_file)

age_col = "subject.ageGroup"
l2_col = "AIFI_L2"
l3_col = "AIFI_L3"
subject_col = "subject.subjectGuid"

obs = adata.obs[[c for c in [subject_col, age_col, l2_col, l3_col] if c in adata.obs.columns]].copy()
obs.to_csv(results_dir / "monocyte_age_composition_input.csv", index=True)

l2_counts = pd.crosstab(obs[age_col], obs[l2_col])
l3_counts = pd.crosstab(obs[age_col], obs[l3_col])

l2_counts.to_csv(results_dir / "monocyte_agegroup_by_AIFI_L2_counts.csv")
l3_counts.to_csv(results_dir / "monocyte_agegroup_by_AIFI_L3_counts.csv")

l2_props = l2_counts.div(l2_counts.sum(axis=1), axis=0)
l3_props = l3_counts.div(l3_counts.sum(axis=1), axis=0)

l2_props.to_csv(results_dir / "monocyte_agegroup_by_AIFI_L2_proportions.csv")
l3_props.to_csv(results_dir / "monocyte_agegroup_by_AIFI_L3_proportions.csv")

subject_l2 = (
    obs.groupby([subject_col, age_col, l2_col])
    .size()
    .reset_index(name="n_cells")
)

subject_totals = (
    obs.groupby([subject_col, age_col])
    .size()
    .reset_index(name="total_cells")
)

subject_l2 = subject_l2.merge(subject_totals, on=[subject_col, age_col], how="left")
subject_l2["proportion"] = subject_l2["n_cells"] / subject_l2["total_cells"]
subject_l2.to_csv(results_dir / "monocyte_subject_level_AIFI_L2_composition.csv", index=False)

subject_l3 = (
    obs.groupby([subject_col, age_col, l3_col])
    .size()
    .reset_index(name="n_cells")
)

subject_l3 = subject_l3.merge(subject_totals, on=[subject_col, age_col], how="left")
subject_l3["proportion"] = subject_l3["n_cells"] / subject_l3["total_cells"]
subject_l3.to_csv(results_dir / "monocyte_subject_level_AIFI_L3_composition.csv", index=False)

print("Done.")
print(f"Saved: {results_dir / 'monocyte_agegroup_by_AIFI_L2_counts.csv'}")
print(f"Saved: {results_dir / 'monocyte_agegroup_by_AIFI_L3_counts.csv'}")
print(f"Saved: {results_dir / 'monocyte_agegroup_by_AIFI_L2_proportions.csv'}")
print(f"Saved: {results_dir / 'monocyte_agegroup_by_AIFI_L3_proportions.csv'}")
print(f"Saved: {results_dir / 'monocyte_subject_level_AIFI_L2_composition.csv'}")
print(f"Saved: {results_dir / 'monocyte_subject_level_AIFI_L3_composition.csv'}")
