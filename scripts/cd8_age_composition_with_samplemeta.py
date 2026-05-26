import scanpy as sc
import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
raw_file = project_dir / "data" / "first_dataset" / "raw" / "human_immune_health_atlas_cd8t-gdt-mait.h5ad"
results_dir = project_dir / "results" / "first_dataset"

results_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading: {raw_file}")
adata = sc.read_h5ad(raw_file)

meta_cols = [
    "subject.subjectGuid",
    "sample.sampleKitGuid",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
    "sample.drawYear",
    "batch_id",
    "AIFI_L2",
    "AIFI_L3",
]

missing = [c for c in meta_cols if c not in adata.obs.columns]
if missing:
    raise ValueError(f"Missing required obs columns: {missing}")

obs = adata.obs[meta_cols].copy()
obs = obs.dropna(subset=[
    "subject.subjectGuid",
    "sample.sampleKitGuid",
    "subject.ageGroup",
    "AIFI_L2",
    "AIFI_L3",
], how="any").copy()

for col in [
    "subject.subjectGuid",
    "sample.sampleKitGuid",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "sample.drawYear",
    "batch_id",
    "AIFI_L2",
    "AIFI_L3",
]:
    obs[col] = obs[col].astype(str)

obs["subject.bmi"] = pd.to_numeric(obs["subject.bmi"], errors="coerce")

group_base = [
    "subject.subjectGuid",
    "sample.sampleKitGuid",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
    "sample.drawYear",
    "batch_id",
]

sample_totals = (
    obs.groupby(group_base, observed=True)
    .size()
    .reset_index(name="total_cells")
)

l2_counts = (
    obs.groupby(group_base + ["AIFI_L2"], observed=True)
    .size()
    .reset_index(name="n_cells")
)
l2_counts = l2_counts.merge(sample_totals, on=group_base, how="left")
l2_counts["proportion"] = l2_counts["n_cells"] / l2_counts["total_cells"]
l2_counts.to_csv(results_dir / "cd8_L2_with_samplemeta.csv", index=False)

l3_counts = (
    obs.groupby(group_base + ["AIFI_L3"], observed=True)
    .size()
    .reset_index(name="n_cells")
)
l3_counts = l3_counts.merge(sample_totals, on=group_base, how="left")
l3_counts["proportion"] = l3_counts["n_cells"] / l3_counts["total_cells"]
l3_counts.to_csv(results_dir / "cd8_L3_with_samplemeta.csv", index=False)

print("Done.")
print(results_dir / "cd8_L2_with_samplemeta.csv")
print(results_dir / "cd8_L3_with_samplemeta.csv")
print("Unique samples in L2:", l2_counts["sample.sampleKitGuid"].nunique())
print("Unique samples in L3:", l3_counts["sample.sampleKitGuid"].nunique())
