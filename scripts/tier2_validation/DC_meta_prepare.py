from pathlib import Path
import pandas as pd

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
meta_dir = outdir

# Load harmonized per-sample summaries already generated
g224 = pd.read_csv(meta_dir / "GSE224273_dc_apc_restricted_sample_summary.csv")
g260 = pd.read_csv(meta_dir / "GSE260657_dc_targeted_sample_summary.csv")
g253 = pd.read_csv(meta_dir / "GSE253902_dc_apc_restricted_sample_summary.csv")

# Standardize dataset names
g224["dataset"] = "GSE224273"
g260["dataset"] = "GSE260657"
g253["dataset"] = "GSE253902"

# Keep harmonized fields only
keep_cols = [
    "dataset", "sample", "status",
    "cdc2_score_mean",
    "prop_cdc2_high",
    "myeloid_score_mean",
]

def prep(df):
    out = df.copy()
    for col in keep_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep_cols].copy()
    return out

combined = pd.concat([prep(g224), prep(g260), prep(g253)], ignore_index=True)

# Keep only asymptomatic/symptomatic labels
combined = combined[combined["status"].isin(["Asymptomatic", "Symptomatic"])].copy()

combined.to_csv(meta_dir / "DC_meta_combined_sample_level.csv", index=False)
print(combined.to_string(index=False))
print("\nSaved:")
print(meta_dir / "DC_meta_combined_sample_level.csv")
