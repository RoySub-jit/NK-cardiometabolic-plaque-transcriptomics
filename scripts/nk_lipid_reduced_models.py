from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

infile = results_dir / "nk_L3_with_clinical_by_sample.csv"
df = pd.read_csv(infile)

wanted_states = [
    "Adaptive NK cell",
    "GZMK- CD56dim NK cell",
    "GZMK+ CD56dim NK cell",
]

predictors = [
    "lip.triglycerides",
    "lip.cholesterol_non_hdl",
]

df = df[df["AIFI_L3"].isin(wanted_states)].copy()

required = [
    "AIFI_L3", "proportion",
    "subject.ageGroup", "subject.biologicalSex", "subject.cmv",
    "subject.bmi", "sample.drawYear",
] + predictors
df = df.dropna(subset=required).copy()

# Types
for col in ["subject.ageGroup", "subject.biologicalSex", "subject.cmv", "sample.drawYear"]:
    df[col] = df[col].astype(str)

df["subject.bmi"] = pd.to_numeric(df["subject.bmi"], errors="coerce")
df["lip.triglycerides"] = pd.to_numeric(df["lip.triglycerides"], errors="coerce")
df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")

df = df.dropna(subset=["subject.bmi", "lip.triglycerides", "lip.cholesterol_non_hdl"]).copy()

# Transform skewed triglycerides
df["log_triglycerides"] = np.log1p(df["lip.triglycerides"])

predictor_map = {
    "lip.triglycerides": "log_triglycerides",
    "lip.cholesterol_non_hdl": "Q('lip.cholesterol_non_hdl')",
}

model_formulas = {
    "M1_age_sex_cmv": (
        "{y} ~ {x} + "
        "C(Q('subject.ageGroup')) + "
        "C(Q('subject.biologicalSex')) + "
        "C(Q('subject.cmv'))"
    ),
    "M2_plus_bmi": (
        "{y} ~ {x} + "
        "C(Q('subject.ageGroup')) + "
        "C(Q('subject.biologicalSex')) + "
        "C(Q('subject.cmv')) + "
        "Q('subject.bmi')"
    ),
    "M3_plus_drawYear": (
        "{y} ~ {x} + "
        "C(Q('subject.ageGroup')) + "
        "C(Q('subject.biologicalSex')) + "
        "C(Q('subject.cmv')) + "
        "Q('subject.bmi') + "
        "C(Q('sample.drawYear'))"
    ),
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
        xterm = predictor_map[predictor]

        for model_name, formula_template in model_formulas.items():
            formula = formula_template.format(y="proportion", x=xterm)

            try:
                fit = smf.ols(formula=formula, data=sub).fit()

                if xterm not in fit.params.index:
                    rows.append({
                        "AIFI_L3": state,
                        "predictor": predictor,
                        "model": model_name,
                        "n_rows": len(sub),
                        "beta": np.nan,
                        "p_value": np.nan,
                        "r_squared": fit.rsquared,
                        "note": "predictor term missing",
                    })
                else:
                    rows.append({
                        "AIFI_L3": state,
                        "predictor": predictor,
                        "model": model_name,
                        "n_rows": len(sub),
                        "beta": fit.params[xterm],
                        "p_value": fit.pvalues[xterm],
                        "r_squared": fit.rsquared,
                        "note": "",
                    })
            except Exception as e:
                rows.append({
                    "AIFI_L3": state,
                    "predictor": predictor,
                    "model": model_name,
                    "n_rows": len(sub),
                    "beta": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "note": str(e),
                })

res = pd.DataFrame(rows)

valid = res["p_value"].notna()
res["fdr_global"] = np.nan
if valid.any():
    res.loc[valid, "fdr_global"] = bh_fdr(res.loc[valid, "p_value"].values)

# also FDR within each model
res["fdr_within_model"] = np.nan
for m in res["model"].dropna().unique():
    mask = (res["model"] == m) & res["p_value"].notna()
    if mask.any():
        res.loc[mask, "fdr_within_model"] = bh_fdr(res.loc[mask, "p_value"].values)

res = res.sort_values(["model", "p_value"], ascending=[True, True])
res.to_csv(results_dir / "nk_L3_lipid_reduced_models.csv", index=False)

print("Done.")
print(results_dir / "nk_L3_lipid_reduced_models.csv")
print(res.to_string(index=False))
