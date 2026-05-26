from pathlib import Path
import pandas as pd

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")

cell_summary = pd.read_csv(outdir / "GSE253902_dc_presence_cell_scores.csv", index_col=0)

# Make sure grouping columns are plain strings, not categorical/object mixtures
for col in ["sample", "status", "gsm"]:
    cell_summary[col] = cell_summary[col].astype(str)

sample_summary = (
    cell_summary.groupby(["sample", "status", "gsm"], observed=True)
    .agg(
        n_cells=("sample", "size"),
        cdc2_score_mean=("cdc2_score", "mean"),
        pdc_score_mean=("pdc_score", "mean"),
        myeloid_score_mean=("myeloid_score", "mean"),
        pdc_minus_cdc2_mean=("pdc_minus_cdc2", "mean"),
    )
    .reset_index()
)

sample_summary.to_csv(outdir / "GSE253902_dc_presence_sample_scores_fixed.csv", index=False)

print(sample_summary.to_string(index=False))
print("\nSaved:")
print(outdir / "GSE253902_dc_presence_sample_scores_fixed.csv")
