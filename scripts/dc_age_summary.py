import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

l2_file = results_dir / "dc_subject_level_AIFI_L2_composition_clean.csv"
l3_file = results_dir / "dc_subject_level_AIFI_L3_composition_clean.csv"

l2 = pd.read_csv(l2_file)
l3 = pd.read_csv(l3_file)

def summarize(df, subtype_col, out_prefix):
    df = df[df["subject.ageGroup"].isin(["Young Adult", "Older Adult"])].copy()

    summary = (
        df.groupby(["subject.ageGroup", subtype_col])["proportion"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    summary.to_csv(results_dir / f"{out_prefix}_agegroup_summary.csv", index=False)

    pivot_mean = summary.pivot(index=subtype_col, columns="subject.ageGroup", values="mean").reset_index()
    if "Older Adult" in pivot_mean.columns and "Young Adult" in pivot_mean.columns:
        pivot_mean["delta_Older_minus_Young"] = pivot_mean["Older Adult"] - pivot_mean["Young Adult"]
    pivot_mean.to_csv(results_dir / f"{out_prefix}_agegroup_mean_delta.csv", index=False)

summarize(l2, "AIFI_L2", "dc_L2")
summarize(l3, "AIFI_L3", "dc_L3")

print("Done.")
print(results_dir / "dc_L2_agegroup_summary.csv")
print(results_dir / "dc_L2_agegroup_mean_delta.csv")
print(results_dir / "dc_L3_agegroup_summary.csv")
print(results_dir / "dc_L3_agegroup_mean_delta.csv")
