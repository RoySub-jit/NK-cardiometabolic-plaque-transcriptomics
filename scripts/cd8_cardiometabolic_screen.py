from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

infile = results_dir / "cd8_L3_with_clinical_by_sample.csv"
df = pd.read_csv(infile)

wanted_states = [
    "KLRF1- GZMB+ CD27- EM CD8 T cell",
    "KLRF1+ GZMB+ CD27- EM CD8 T cell",
    "GZMK+ CD27+ EM CD8 T cell",
    "CD8 MAIT",
    "GZMK+ Vd2 gdT",
]

predictors = [
    "lip.cholesterol_non_hdl",
    "infl.hs_crp",
]

df = df[df["AIFI_L3"].isin(wanted_states)].copy()

required = [
    "AIFI_L3", "proportion",
    "subject.ageGroup", "subject.biologicalSex", "subject.cmv", "subject.bmi"
] + predictors

df = df.dropna(subset=required).copy()

for col in ["subject.ageGroup", "subject.biologicalSex", "subject.cmv"]:
    df[col] = df[col].astype(str)

df["subject.bmi"] = pd.to_numeric(df["subject.bmi"], errors="coerce")
df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")
df["infl.hs_crp"] = pd.to_numeric(df["infl.hs_crp"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")

df = df.dropna(subset=["subject.bmi", "lip.cholesterol_non_hdl", "infl.hs_crp"]).copy()
df["log_hs_crp"] = np.log1p(df["infl.hs_crp"])

predictor_map = {
    "lip.cholesterol_non_hdl": "Q('lip.cholesterol_non_hdl')",
    "infl.hs_crp": "log_hs_crp",
}

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out

rows = []

for state in wanted_states:
    sub = df[df["AIFI_L3"] == state].copy()

    for predictor in predictors:
        formula = (
            "proportion ~ "
            f"{predictor_map[predictor]} + "
            "C(Q('subject.ageGroup')) + "
            "C(Q('subject.biologicalSex')) + "
            "C(Q('subject.cmv')) + "
            "Q('subject.bmi')"
        )

        try:
            fit = smf.ols(formula=formula, data=sub).fit()
            term = predictor_map[predictor]
            rows.append({
                "AIFI_L3": state,
                "predictor": predictor,
                "n_rows": len(sub),
                "beta": fit.params.get(term, np.nan),
                "p_value": fit.pvalues.get(term, np.nan),
                "r_squared": fit.rsquared,
                "note": "",
            })
        except Exception as e:
            rows.append({
                "AIFI_L3": state,
                "predictor": predictor,
                "n_rows": len(sub),
                "beta": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "note": str(e),
            })

res = pd.DataFrame(rows)
valid = res["p_value"].notna()
res["fdr"] = np.nan
if valid.any():
    res.loc[valid, "fdr"] = bh_fdr(res.loc[valid, "p_value"].values)

res = res.sort_values(["fdr", "p_value"], ascending=[True, True])
res.to_csv(results_dir / "cd8_L3_cardiometabolic_screen.csv", index=False)

print("Done.")
print(results_dir / "cd8_L3_cardiometabolic_screen.csv")
print(res.to_string(index=False))
