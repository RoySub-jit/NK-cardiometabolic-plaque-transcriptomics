from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"

rng = np.random.default_rng(42)

def fit_model(df, formula, predictor_term):
    fit = smf.ols(formula=formula, data=df).fit()
    return {
        "beta": fit.params.get(predictor_term, np.nan),
        "std_err": fit.bse.get(predictor_term, np.nan),
        "t_value": fit.tvalues.get(predictor_term, np.nan),
        "p_value": fit.pvalues.get(predictor_term, np.nan),
        "conf_low": fit.conf_int().loc[predictor_term, 0] if predictor_term in fit.params.index else np.nan,
        "conf_high": fit.conf_int().loc[predictor_term, 1] if predictor_term in fit.params.index else np.nan,
        "r_squared": fit.rsquared,
        "adj_r_squared": fit.rsquared_adj,
    }

def leave_one_out(df, sample_col, formula, predictor_term, expected_sign=None):
    rows = []
    samples = df[sample_col].dropna().unique().tolist()

    for s in samples:
        sub = df[df[sample_col] != s].copy()
        try:
            fit = smf.ols(formula=formula, data=sub).fit()
            beta = fit.params.get(predictor_term, np.nan)
            p = fit.pvalues.get(predictor_term, np.nan)
            rows.append({
                "left_out_sample": s,
                "n_rows": len(sub),
                "beta": beta,
                "p_value": p,
                "r_squared": fit.rsquared,
            })
        except Exception as e:
            rows.append({
                "left_out_sample": s,
                "n_rows": len(sub),
                "beta": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
            })

    out = pd.DataFrame(rows)

    sign_positive = (out["beta"] > 0).mean()
    sign_negative = (out["beta"] < 0).mean()

    summary = {
        "n_leave_one_out_models": len(out),
        "beta_median": out["beta"].median(),
        "beta_min": out["beta"].min(),
        "beta_max": out["beta"].max(),
        "p_median": out["p_value"].median(),
        "p_min": out["p_value"].min(),
        "p_max": out["p_value"].max(),
        "prop_beta_positive": sign_positive,
        "prop_beta_negative": sign_negative,
    }

    if expected_sign == "positive":
        summary["prop_expected_sign"] = sign_positive
    elif expected_sign == "negative":
        summary["prop_expected_sign"] = sign_negative
    else:
        summary["prop_expected_sign"] = np.nan

    return out, summary

def bootstrap_sign_stability(df, formula, predictor_term, sample_col, expected_sign=None, n_boot=1000):
    rows = []
    samples = df[sample_col].dropna().unique().tolist()
    n = len(samples)

    # bootstrap by sample ID
    for i in range(n_boot):
        boot_ids = rng.choice(samples, size=n, replace=True)
        boot_df = pd.concat([df[df[sample_col] == s] for s in boot_ids], ignore_index=True)

        try:
            fit = smf.ols(formula=formula, data=boot_df).fit()
            beta = fit.params.get(predictor_term, np.nan)
            p = fit.pvalues.get(predictor_term, np.nan)
            rows.append({"iter": i + 1, "beta": beta, "p_value": p})
        except Exception:
            rows.append({"iter": i + 1, "beta": np.nan, "p_value": np.nan})

    out = pd.DataFrame(rows)

    summary = {
        "n_bootstrap": len(out),
        "beta_median": out["beta"].median(),
        "beta_mean": out["beta"].mean(),
        "beta_ci_low_2_5": out["beta"].quantile(0.025),
        "beta_ci_high_97_5": out["beta"].quantile(0.975),
        "p_median": out["p_value"].median(),
        "prop_p_lt_0_05": (out["p_value"] < 0.05).mean(),
        "prop_beta_positive": (out["beta"] > 0).mean(),
        "prop_beta_negative": (out["beta"] < 0).mean(),
    }

    if expected_sign == "positive":
        summary["prop_expected_sign"] = (out["beta"] > 0).mean()
    elif expected_sign == "negative":
        summary["prop_expected_sign"] = (out["beta"] < 0).mean()
    else:
        summary["prop_expected_sign"] = np.nan

    return out, summary

# -----------------------------
# NK robustness checks
# -----------------------------
nk_file = results_dir / "nk_L3_with_clinical_by_sample.csv"
nk = pd.read_csv(nk_file)
nk = nk[nk["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()

nk["subject.bmi"] = pd.to_numeric(nk["subject.bmi"], errors="coerce")
nk["lip.cholesterol_non_hdl"] = pd.to_numeric(nk["lip.cholesterol_non_hdl"], errors="coerce")
nk["subject.ageAtFirstDraw"] = pd.to_numeric(nk["subject.ageAtFirstDraw"], errors="coerce")
nk["proportion"] = pd.to_numeric(nk["proportion"], errors="coerce")

for col in ["subject.ageGroup", "subject.biologicalSex", "subject.cmv"]:
    nk[col] = nk[col].astype(str)

nk = nk.dropna(subset=[
    "sample.sampleKitGuid",
    "proportion",
    "lip.cholesterol_non_hdl",
    "subject.ageGroup",
    "subject.ageAtFirstDraw",
    "subject.biologicalSex",
    "subject.cmv",
    "subject.bmi",
]).copy()

nk_formula_age_cont = (
    "proportion ~ Q('lip.cholesterol_non_hdl') + "
    "Q('subject.ageAtFirstDraw') + "
    "C(Q('subject.biologicalSex')) + "
    "C(Q('subject.cmv')) + "
    "Q('subject.bmi')"
)

nk_predictor = "Q('lip.cholesterol_non_hdl')"

nk_age_cont = pd.DataFrame([{
    "analysis": "NK_age_continuous",
    **fit_model(nk, nk_formula_age_cont, nk_predictor),
    "n_samples": nk["sample.sampleKitGuid"].nunique(),
}])
nk_age_cont.to_csv(results_dir / "nk_primary_nonhdl_age_continuous_summary.csv", index=False)

nk_loo_df, nk_loo_summary = leave_one_out(
    nk, "sample.sampleKitGuid", nk_formula_age_cont, nk_predictor, expected_sign="positive"
)
nk_loo_df.to_csv(results_dir / "nk_primary_nonhdl_leave_one_out_details.csv", index=False)
pd.DataFrame([{"analysis": "NK_leave_one_out", **nk_loo_summary}]).to_csv(
    results_dir / "nk_primary_nonhdl_leave_one_out_summary.csv", index=False
)

nk_boot_df, nk_boot_summary = bootstrap_sign_stability(
    nk, nk_formula_age_cont, nk_predictor, "sample.sampleKitGuid", expected_sign="positive", n_boot=1000
)
nk_boot_df.to_csv(results_dir / "nk_primary_nonhdl_bootstrap_details.csv", index=False)
pd.DataFrame([{"analysis": "NK_bootstrap", **nk_boot_summary}]).to_csv(
    results_dir / "nk_primary_nonhdl_bootstrap_summary.csv", index=False
)

# -----------------------------
# DC robustness checks
# -----------------------------
dc_file = results_dir / "dc_L3_with_clinical_by_sample.csv"
dc = pd.read_csv(dc_file)
dc = dc[dc["AIFI_L3"] == "HLA-DRhi cDC2"].copy()

dc["subject.bmi"] = pd.to_numeric(dc["subject.bmi"], errors="coerce")
dc["infl.hs_crp"] = pd.to_numeric(dc["infl.hs_crp"], errors="coerce")
dc["subject.ageAtFirstDraw"] = pd.to_numeric(dc["subject.ageAtFirstDraw"], errors="coerce")
dc["proportion"] = pd.to_numeric(dc["proportion"], errors="coerce")
dc["log_hs_crp"] = np.log1p(dc["infl.hs_crp"])

for col in ["subject.ageGroup", "subject.biologicalSex"]:
    dc[col] = dc[col].astype(str)

dc = dc.dropna(subset=[
    "sample.sampleKitGuid",
    "proportion",
    "infl.hs_crp",
    "subject.ageGroup",
    "subject.ageAtFirstDraw",
    "subject.biologicalSex",
    "subject.bmi",
]).copy()

dc_formula_age_cont = (
    "proportion ~ log_hs_crp + "
    "Q('subject.ageAtFirstDraw') + "
    "C(Q('subject.biologicalSex')) + "
    "Q('subject.bmi')"
)

dc_predictor = "log_hs_crp"

dc_age_cont = pd.DataFrame([{
    "analysis": "DC_age_continuous",
    **fit_model(dc, dc_formula_age_cont, dc_predictor),
    "n_samples": dc["sample.sampleKitGuid"].nunique(),
}])
dc_age_cont.to_csv(results_dir / "dc_primary_hscrp_age_continuous_summary.csv", index=False)

dc_loo_df, dc_loo_summary = leave_one_out(
    dc, "sample.sampleKitGuid", dc_formula_age_cont, dc_predictor, expected_sign="negative"
)
dc_loo_df.to_csv(results_dir / "dc_primary_hscrp_leave_one_out_details.csv", index=False)
pd.DataFrame([{"analysis": "DC_leave_one_out", **dc_loo_summary}]).to_csv(
    results_dir / "dc_primary_hscrp_leave_one_out_summary.csv", index=False
)

dc_boot_df, dc_boot_summary = bootstrap_sign_stability(
    dc, dc_formula_age_cont, dc_predictor, "sample.sampleKitGuid", expected_sign="negative", n_boot=1000
)
dc_boot_df.to_csv(results_dir / "dc_primary_hscrp_bootstrap_details.csv", index=False)
pd.DataFrame([{"analysis": "DC_bootstrap", **dc_boot_summary}]).to_csv(
    results_dir / "dc_primary_hscrp_bootstrap_summary.csv", index=False
)

# -----------------------------
# Combined summary
# -----------------------------
combined = pd.concat([
    nk_age_cont.assign(group="age_continuous"),
    pd.DataFrame([{"analysis": "NK_leave_one_out", **nk_loo_summary, "group": "leave_one_out"}]),
    pd.DataFrame([{"analysis": "NK_bootstrap", **nk_boot_summary, "group": "bootstrap"}]),
    dc_age_cont.assign(group="age_continuous"),
    pd.DataFrame([{"analysis": "DC_leave_one_out", **dc_loo_summary, "group": "leave_one_out"}]),
    pd.DataFrame([{"analysis": "DC_bootstrap", **dc_boot_summary, "group": "bootstrap"}]),
], ignore_index=True)

combined.to_csv(results_dir / "primary_robustness_checks_combined_summary.csv", index=False)

print("Done.")
print(results_dir / "nk_primary_nonhdl_age_continuous_summary.csv")
print(results_dir / "nk_primary_nonhdl_leave_one_out_summary.csv")
print(results_dir / "nk_primary_nonhdl_bootstrap_summary.csv")
print(results_dir / "dc_primary_hscrp_age_continuous_summary.csv")
print(results_dir / "dc_primary_hscrp_leave_one_out_summary.csv")
print(results_dir / "dc_primary_hscrp_bootstrap_summary.csv")
print(results_dir / "primary_robustness_checks_combined_summary.csv")
print("\nCombined summary:")
print(combined.to_string(index=False))
