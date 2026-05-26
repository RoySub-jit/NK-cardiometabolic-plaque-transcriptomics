import scanpy as sc
import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
raw_file = project_dir / "data" / "first_dataset" / "raw" / "human_immune_health_atlas_mono.h5ad"
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"

results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

sc.settings.figdir = str(figures_dir)
sc.settings.verbosity = 3

print(f"Reading: {raw_file}")
adata = sc.read_h5ad(raw_file)

print("\nADATA:")
print(adata)

print("\nOBS columns:")
print(list(adata.obs.columns))

# Save metadata summary
obs_cols = [
    "subject.subjectGuid",
    "subject.biologicalSex",
    "subject.ageAtFirstDraw",
    "subject.ageGroup",
    "sample.subjectAgeAtDraw",
    "subject.cmv",
    "subject.bmi",
    "AIFI_L1",
    "AIFI_L2",
    "AIFI_L3"
]

existing_cols = [c for c in obs_cols if c in adata.obs.columns]
meta = adata.obs[existing_cols].copy()
meta.to_csv(results_dir / "monocyte_metadata_preview.csv", index=True)

# Basic summaries
summary_lines = []

summary_lines.append(f"Cells: {adata.n_obs}")
summary_lines.append(f"Genes: {adata.n_vars}")

for col in ["subject.ageGroup", "subject.biologicalSex", "subject.cmv", "AIFI_L1", "AIFI_L2", "AIFI_L3"]:
    if col in adata.obs.columns:
        vc = adata.obs[col].astype(str).value_counts(dropna=False)
        vc.to_csv(results_dir / f"{col.replace('.', '_')}_counts.csv")
        summary_lines.append(f"\n{col} counts:\n{vc.to_string()}")

with open(results_dir / "monocyte_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

# Plot existing UMAP colored by key metadata
plot_cols = [c for c in ["subject.ageGroup", "subject.biologicalSex", "AIFI_L1", "AIFI_L2", "AIFI_L3"] if c in adata.obs.columns]

for col in plot_cols:
    sc.pl.umap(adata, color=col, save=f"_mono_{col.replace('.', '_')}.png", show=False)

# Save a lighter working file with only major metadata retained
keep_cols = [c for c in [
    "subject.subjectGuid",
    "subject.biologicalSex",
    "subject.ageAtFirstDraw",
    "subject.ageGroup",
    "sample.subjectAgeAtDraw",
    "subject.cmv",
    "subject.bmi",
    "AIFI_L1",
    "AIFI_L2",
    "AIFI_L3"
] if c in adata.obs.columns]

obs_export = adata.obs[keep_cols].copy()
obs_export.to_csv(results_dir / "monocyte_obs_selected.csv", index=True)

print("\nDone.")
print(f"Saved summary: {results_dir / 'monocyte_summary.txt'}")
print(f"Saved metadata preview: {results_dir / 'monocyte_metadata_preview.csv'}")
print(f"Saved selected obs table: {results_dir / 'monocyte_obs_selected.csv'}")
print(f"Saved figures in: {figures_dir}")
