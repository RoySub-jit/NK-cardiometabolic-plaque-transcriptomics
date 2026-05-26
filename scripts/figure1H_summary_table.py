from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

# ---------- Load primary model summaries ----------
nk = pd.read_csv(results_dir / "nk_primary_nonhdl_model_summary.csv")
dc = pd.read_csv(results_dir / "dc_primary_hscrp_model_summary.csv")
cd8 = pd.read_csv(results_dir / "cd8_primary_hscrp_gzmk_vd2gdt_summary.csv")

# ---------- Load model inputs ----------
nk_input = pd.read_csv(results_dir / "nk_primary_nonhdl_model_input.csv")
dc_input = pd.read_csv(results_dir / "dc_primary_hscrp_model_input.csv")
cd8_input = pd.read_csv(results_dir / "cd8_primary_hscrp_gzmk_vd2gdt_summary.csv")  # not input, fix below if needed
# Correct CD8 input path if available
cd8_input_file = results_dir / "cd8_primary_hscrp_gzmk_vd2gdt_input.csv"
if cd8_input_file.exists():
    cd8_input = pd.read_csv(cd8_input_file)
else:
    # fallback: reconstruct from merged table
    merged = pd.read_csv(results_dir / "cd8_L3_with_clinical_by_sample.csv")
    cd8_input = merged[merged["AIFI_L3"] == "GZMK+ Vd2 gdT"].copy()
    cd8_input = cd8_input[[
        "sample.sampleKitGuid", "subject.subjectGuid", "proportion",
        "subject.ageGroup", "subject.biologicalSex", "subject.bmi", "infl.hs_crp"
    ]].copy()
    cd8_input["subject.bmi"] = pd.to_numeric(cd8_input["subject.bmi"], errors="coerce")
    cd8_input["infl.hs_crp"] = pd.to_numeric(cd8_input["infl.hs_crp"], errors="coerce")
    cd8_input["proportion"] = pd.to_numeric(cd8_input["proportion"], errors="coerce")
    cd8_input = cd8_input.dropna(subset=["proportion", "infl.hs_crp", "subject.ageGroup", "subject.biologicalSex", "subject.bmi"]).copy()

def kruskal_p_from_tertiles(df, predictor_col, value_col, labels):
    tmp = df[[predictor_col, value_col]].dropna().copy()
    tmp["tertile"] = pd.qcut(tmp[predictor_col], q=3, labels=labels, duplicates="drop")
    groups = [tmp.loc[tmp["tertile"] == lab, value_col].dropna().values for lab in labels if (tmp["tertile"] == lab).any()]
    if len(groups) < 2:
        return np.nan
    return kruskal(*groups).pvalue

# ---------- Compute tertile/boxplot p-values ----------
nk_box_p = kruskal_p_from_tertiles(
    nk_input, "lip.cholesterol_non_hdl", "proportion",
    ["Low non-HDL", "Mid non-HDL", "High non-HDL"]
)

dc_box_p = kruskal_p_from_tertiles(
    dc_input, "infl.hs_crp", "proportion",
    ["Low hs-CRP", "Mid hs-CRP", "High hs-CRP"]
)

cd8_box_p = kruskal_p_from_tertiles(
    cd8_input, "infl.hs_crp", "proportion",
    ["Low hs-CRP", "Mid hs-CRP", "High hs-CRP"]
)

# ---------- Build summary table ----------
table = pd.DataFrame([
    {
        "panel": "B-C",
        "lineage": "NK",
        "outcome": nk.loc[0, "outcome"],
        "predictor": nk.loc[0, "predictor"],
        "boxplot_test": "Kruskal-Wallis across tertiles",
        "boxplot_p_value": nk_box_p,
        "beta": nk.loc[0, "beta"],
        "model_p_value": nk.loc[0, "p_value"],
        "r_squared": nk.loc[0, "r_squared"],
        "adj_r_squared": nk.loc[0, "adj_r_squared"],
        "n_samples": nk.loc[0, "n_samples"],
    },
    {
        "panel": "D-E",
        "lineage": "DC",
        "outcome": dc.loc[0, "outcome"],
        "predictor": dc.loc[0, "predictor"],
        "boxplot_test": "Kruskal-Wallis across tertiles",
        "boxplot_p_value": dc_box_p,
        "beta": dc.loc[0, "beta"],
        "model_p_value": dc.loc[0, "p_value"],
        "r_squared": dc.loc[0, "r_squared"],
        "adj_r_squared": dc.loc[0, "adj_r_squared"],
        "n_samples": dc.loc[0, "n_samples"],
    },
    {
        "panel": "F-G",
        "lineage": "CD8/gdT",
        "outcome": cd8.loc[0, "outcome"],
        "predictor": cd8.loc[0, "predictor"],
        "boxplot_test": "Kruskal-Wallis across tertiles",
        "boxplot_p_value": cd8_box_p,
        "beta": cd8.loc[0, "beta"],
        "model_p_value": cd8.loc[0, "p_value"],
        "r_squared": cd8.loc[0, "r_squared"],
        "adj_r_squared": cd8.loc[0, "adj_r_squared"],
        "n_samples": cd8.loc[0, "n_samples"],
    },
])

table.to_csv(results_dir / "Figure1H_summary_table.csv", index=False)
print("Done.")
print(results_dir / "Figure1H_summary_table.csv")
print(table.to_string(index=False))
