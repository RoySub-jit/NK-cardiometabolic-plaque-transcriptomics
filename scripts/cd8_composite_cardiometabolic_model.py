from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

infile = results_dir / "cd8_L3_with_clinical_by_sample.csv"
df = pd.read_csv(infile)

states = [
    "KLRF1+ GZMB+ CD27- EM CD8 T cell",
    "KLRF1- GZMB+ CD27- EM CD8 T cell",
    "GZMK+ Vd2 gdT",
    "CD8 MAIT",
]

keep = [
    "sample.sampleKitGuid",
    "subject.subjectGuid",
    "AIFI_L3",
    "proportion",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.bmi",
    "infl.hs_crp",
    "lip.cholesterol_non_hdl",
]
df = df[keep].copy()
df = df[df["AIFI_L3"].isin(states)].copy()

# Pivot to sample-level wide table
wide = df.pivot_table(
    index=[
        "sample.sampleKitGuid",
        "subject.subjectGuid",
        "subject.ageGroup",
        "subject.biologicalSex",
        "subject.bmi",
        "infl.hs_crp",
        "lip.cholesterol_non_hdl",
    ],
    columns="AIFI_L3",
    values="proportion",
    aggfunc="first",
).reset_index()

# Ensure all composite states exist
for s in states:
    if s not in wide.columns:
        wide[s] = 0.0
    wide[s] = pd.to_numeric(wide[s], errors="coerce").fillna(0.0)

# Build composite
wide["cyto_innate_t_composite"] = wide[states].sum(axis=1)

# Clean metadata
for col in ["subject.ageGroup", "subject.biologicalSex"]:
    wide[col] = wide[col].astype(str)

wide["subject.bmi"] = pd.to_numeric(wide["subject.bmi"], errors="coerce")
wide["infl.hs_crp"] = pd.to_numeric(wide["infl.hs_crp"], errors="coerce")
wide["lip.cholesterol_non_hdl"] = pd.to_numeric(wide["lip.cholesterol_non_hdl"], errors="coerce")
wide = wide.dropna(subset=["subject.bmi", "infl.hs_crp", "lip.cholesterol_non_hdl"]).copy()

# Log-transform hs-CRP
wide["log_hs_crp"] = np.log1p(wide["infl.hs_crp"])

# Primary composite model with hs-CRP
formula_hscrp = (
    "cyto_innate_t_composite ~ log_hs_crp + "
    "C(Q('subject.ageGroup')) + "
    "C(Q('subject.biologicalSex')) + "
    "Q('subject.bmi')"
)
fit_hscrp = smf.ols(formula=formula_hscrp, data=wide).fit()

# Secondary composite model with non-HDL
formula_nonhdl = (
    "cyto_innate_t_composite ~ Q('lip.cholesterol_non_hdl') + "
    "C(Q('subject.ageGroup')) + "
    "C(Q('subject.biologicalSex')) + "
    "Q('subject.bmi')"
)
fit_nonhdl = smf.ols(formula=formula_nonhdl, data=wide).fit()

out = pd.DataFrame([
    {
        "outcome": "cyto_innate_t_composite",
        "predictor": "infl.hs_crp",
        "n_samples": int(wide["sample.sampleKitGuid"].nunique()),
        "beta": fit_hscrp.params.get("log_hs_crp", np.nan),
        "std_err": fit_hscrp.bse.get("log_hs_crp", np.nan),
        "t_value": fit_hscrp.tvalues.get("log_hs_crp", np.nan),
        "p_value": fit_hscrp.pvalues.get("log_hs_crp", np.nan),
        "conf_low": fit_hscrp.conf_int().loc["log_hs_crp", 0] if "log_hs_crp" in fit_hscrp.params.index else np.nan,
        "conf_high": fit_hscrp.conf_int().loc["log_hs_crp", 1] if "log_hs_crp" in fit_hscrp.params.index else np.nan,
        "r_squared": fit_hscrp.rsquared,
        "adj_r_squared": fit_hscrp.rsquared_adj,
    },
    {
        "outcome": "cyto_innate_t_composite",
        "predictor": "lip.cholesterol_non_hdl",
        "n_samples": int(wide["sample.sampleKitGuid"].nunique()),
        "beta": fit_nonhdl.params.get("Q('lip.cholesterol_non_hdl')", np.nan),
        "std_err": fit_nonhdl.bse.get("Q('lip.cholesterol_non_hdl')", np.nan),
        "t_value": fit_nonhdl.tvalues.get("Q('lip.cholesterol_non_hdl')", np.nan),
        "p_value": fit_nonhdl.pvalues.get("Q('lip.cholesterol_non_hdl')", np.nan),
        "conf_low": fit_nonhdl.conf_int().loc["Q('lip.cholesterol_non_hdl')", 0] if "Q('lip.cholesterol_non_hdl')" in fit_nonhdl.params.index else np.nan,
        "conf_high": fit_nonhdl.conf_int().loc["Q('lip.cholesterol_non_hdl')", 1] if "Q('lip.cholesterol_non_hdl')" in fit_nonhdl.params.index else np.nan,
        "r_squared": fit_nonhdl.rsquared,
        "adj_r_squared": fit_nonhdl.rsquared_adj,
    },
])

out.to_csv(results_dir / "cd8_composite_cardiometabolic_summary.csv", index=False)
wide.to_csv(results_dir / "cd8_composite_cardiometabolic_input.csv", index=False)

with open(results_dir / "cd8_composite_hscrp_full_summary.txt", "w") as f:
    f.write(fit_hscrp.summary().as_text())

with open(results_dir / "cd8_composite_nonhdl_full_summary.txt", "w") as f:
    f.write(fit_nonhdl.summary().as_text())

# hs-CRP scatter
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.scatter(wide["infl.hs_crp"], wide["cyto_innate_t_composite"], alpha=0.8, s=28)

x = np.linspace(wide["infl.hs_crp"].min(), wide["infl.hs_crp"].max(), 100)
plot_df = pd.DataFrame({
    "log_hs_crp": np.log1p(x),
    "subject.ageGroup": [wide["subject.ageGroup"].mode().iloc[0]] * len(x),
    "subject.biologicalSex": [wide["subject.biologicalSex"].mode().iloc[0]] * len(x),
    "subject.bmi": [wide["subject.bmi"].median()] * len(x),
})
ax.plot(x, fit_hscrp.predict(plot_df), linewidth=2)

beta = out.loc[out["predictor"] == "infl.hs_crp", "beta"].iloc[0]
pval = out.loc[out["predictor"] == "infl.hs_crp", "p_value"].iloc[0]
ax.set_xlabel("hs-CRP")
ax.set_ylabel("Cytotoxic / innate-like T-cell composite")
ax.set_title(f"Composite vs hs-CRP\nβ={beta:.5f}, p={pval:.4f}")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_cd8_composite_hscrp_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_cd8_composite_hscrp_scatter.pdf", bbox_inches="tight")
plt.close(fig)

# non-HDL scatter
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.scatter(wide["lip.cholesterol_non_hdl"], wide["cyto_innate_t_composite"], alpha=0.8, s=28)

x = np.linspace(wide["lip.cholesterol_non_hdl"].min(), wide["lip.cholesterol_non_hdl"].max(), 100)
plot_df = pd.DataFrame({
    "lip.cholesterol_non_hdl": x,
    "subject.ageGroup": [wide["subject.ageGroup"].mode().iloc[0]] * len(x),
    "subject.biologicalSex": [wide["subject.biologicalSex"].mode().iloc[0]] * len(x),
    "subject.bmi": [wide["subject.bmi"].median()] * len(x),
})
ax.plot(x, fit_nonhdl.predict(plot_df), linewidth=2)

beta = out.loc[out["predictor"] == "lip.cholesterol_non_hdl", "beta"].iloc[0]
pval = out.loc[out["predictor"] == "lip.cholesterol_non_hdl", "p_value"].iloc[0]
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("Cytotoxic / innate-like T-cell composite")
ax.set_title(f"Composite vs non-HDL\nβ={beta:.5f}, p={pval:.4f}")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_cd8_composite_nonhdl_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_cd8_composite_nonhdl_scatter.pdf", bbox_inches="tight")
plt.close(fig)

print("Done.")
print(results_dir / "cd8_composite_cardiometabolic_summary.csv")
print(figures_dir / "Figure_cd8_composite_hscrp_scatter.png")
print(figures_dir / "Figure_cd8_composite_nonhdl_scatter.png")
