import scanpy as sc
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
raw_file = project_dir / "data" / "first_dataset" / "raw" / "human_immune_health_atlas_nk-ilc.h5ad"

print(f"Reading: {raw_file}")
adata = sc.read_h5ad(raw_file)

print("\nADATA:")
print(adata)

print("\nFirst 60 obs columns:")
print(list(adata.obs.columns)[:60])

print("\nFirst 30 var columns:")
print(list(adata.var.columns)[:30])

for col in ["subject.ageGroup", "subject.subjectGuid", "AIFI_L1", "AIFI_L2", "AIFI_L3"]:
    if col in adata.obs.columns:
        print(f"\n{col} value counts:")
        print(adata.obs[col].astype(str).value_counts(dropna=False).head(30))
