from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

infile = results_dir / "nk_L3_with_clinical_by_sample.csv"
df = pd.read_csv(infile)

state = "GZMK+ CD56dim NK cell"

keep = [
    "AIFI_L3",
    "proportion",
    "sample.sampleKitGuid",
    "subject.subjectGuid",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
    "lip.cholesterol_non_hdl",
]
df = df[keep].copy()
df = df[df["AIFI_L3"] == state].copy()

# clean types
for col in ["subject.ageGroup", "subject.biologicalSex", "subject.cmv"]:
    df[col] = df[col].astype(str)

df["subject.bmi"] = pd.to_numeric(df["subject.bmi"], errors="coerce")
df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")

df = df.dropna(subset=[
    "proportion",
    "lip.cholesterol_non_hdl",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
]).copy()

rows = []

def fit_and_record(sub, group_type, group_value, formula):
    term = "Q('lip.cholesterol_non_hdl')"
    n_samples = sub["sample.sampleKitGuid"].nunique()

    try:
        fit = smf.ols(formula=formula, data=sub).fit()
        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "n_samples": n_samples,
            "beta": fit.params.get(term, np.nan),
            "std_err": fit.bse.get(term, np.nan),
            "t_value": fit.tvalues.get(term, np.nan),
            "p_value": fit.pvalues.get(term, np.nan),
            "conf_low": fit.conf_int().loc[term, 0] if term in fit.params.index else np.nan,
            "conf_high": fit.conf_int().loc[term, 1] if term in fit.params.index else np.nan,
            "r_squared": fit.rsquared,
            "adj_r_squared": fit.rsquared_adj,
            "note": "",
        })
    except Exception as e:
        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "n_samples": n_samples,
            "beta": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "note": str(e),
        })

# CMV-stratified
for cmv_value in sorted(df["subject.cmv"].dropna().unique()):
    sub = df[df["subject.cmv"] == cmv_value].copy()
    if sub["sample.sampleKitGuid"].nunique() < 15:
        rows.append({
            "group_type": "CMV",
            "group_value": cmv_value,
            "n_samples": sub["sample.sampleKitGuid"].nunique(),
            "beta": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "note": "too few samples",
        })
        continue

    formula = (
        "proportion ~ Q('lip.cholesterol_non_hdl') + "
        "C(Q('subject.ageGroup')) + "
        "C(Q('subject.biologicalSex')) + "
        "Q('subject.bmi')"
    )
    fit_and_record(sub, "CMV", cmv_value, formula)

# Sex-stratified
for sex_value in sorted(df["subject.biologicalSex"].dropna().unique()):
    sub = df[df["subject.biologicalSex"] == sex_value].copy()
    if sub["sample.sampleKitGuid"].nunique() < 15:
        rows.append({
            "group_type": "Sex",
            "group_value": sex_value,
            "n_samples": sub["sample.sampleKitGuid"].nunique(),
            "beta": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "note": "too few samples",
        })
        continue

    formula = (
        "proportion ~ Q('lip.cholesterol_non_hdl') + "
        "C(Q('subject.ageGroup')) + "
        "C(Q('subject.cmv')) + "
        "Q('subject.bmi')"
    )
    fit_and_record(sub, "Sex", sex_value, formula)

res = pd.DataFrame(rows)
res.to_csv(results_dir / "nk_primary_nonhdl_stratified_summary.csv", index=False)

print("Done.")
print(results_dir / "nk_primary_nonhdl_stratified_summary.csv")
print(res.to_string(index=False))
