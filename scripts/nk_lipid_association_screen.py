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
    "ISG+ CD56dim NK cell",
    "Proliferating NK cell",
]

predictors = [
    "lip.triglycerides",
    "lip.cholesterol_non_hdl",
    "lip.cholesterol_hdl",
    "infl.hs_crp",
]

covariates = [
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
    "sample.drawYear",
    "batch_id",
]

df = df[df["AIFI_L3"].isin(wanted_states)].copy()

required = ["AIFI_L3", "proportion"] + predictors + covariates
df = df.dropna(subset=required).copy()

df["subject.ageGroup"] = df["subject.ageGroup"].astype(str)
df["subject.biologicalSex"] = df["subject.biologicalSex"].astype(str)
df["subject.cmv"] = df["subject.cmv"].astype(str)
df["sample.drawYear"] = df["sample.drawYear"].astype(str)
df["batch_id"] = df["batch_id"].astype(str)
df["subject.bmi"] = pd.to_numeric(df["subject.bmi"], errors="coerce")

for p in predictors:
    df[p] = pd.to_numeric(df[p], errors="coerce")

df = df.dropna(subset=["subject.bmi"] + predictors).copy()

df["log_triglycerides"] = np.log1p(df["lip.triglycerides"])
df["log_hs_crp"] = np.log1p(df["infl.hs_crp"])

predictor_map = {
    "lip.triglycerides": "log_triglycerides",
    "lip.cholesterol_non_hdl": "Q('lip.cholesterol_non_hdl')",
    "lip.cholesterol_hdl": "Q('lip.cholesterol_hdl')",
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
            "Q('subject.bmi') + "
            "C(Q('sample.drawYear')) + "
            "C(Q('batch_id'))"
        )

        try:
            model = smf.ols(formula=formula, data=sub).fit()
            term = predictor_map[predictor]

            if term not in model.params.index:
                rows.append({
                    "AIFI_L3": state,
                    "predictor": predictor,
                    "n_rows": len(sub),
                    "beta": np.nan,
                    "p_value": np.nan,
                    "r_squared": model.rsquared,
                    "note": "predictor term missing in model",
                })
            else:
                rows.append({
                    "AIFI_L3": state,
                    "predictor": predictor,
                    "n_rows": len(sub),
                    "beta": model.params[term],
                    "p_value": model.pvalues[term],
                    "r_squared": model.rsquared,
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
res.to_csv(results_dir / "nk_L3_lipid_association_screen.csv", index=False)

print("Done.")
print(results_dir / "nk_L3_lipid_association_screen.csv")
print(res.to_string(index=False))
