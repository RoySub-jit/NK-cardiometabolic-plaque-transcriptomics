from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

infile = results_dir / "dc_L3_with_clinical_by_sample.csv"
df = pd.read_csv(infile)

state = "HLA-DRhi cDC2"

keep = [
    "AIFI_L3",
    "proportion",
    "sample.sampleKitGuid",
    "subject.subjectGuid",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.bmi",
    "infl.hs_crp",
]
df = df[keep].copy()
df = df[df["AIFI_L3"] == state].copy()

for col in ["subject.ageGroup", "subject.biologicalSex"]:
    df[col] = df[col].astype(str)

df["subject.bmi"] = pd.to_numeric(df["subject.bmi"], errors="coerce")
df["infl.hs_crp"] = pd.to_numeric(df["infl.hs_crp"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")

df = df.dropna(subset=[
    "proportion",
    "infl.hs_crp",
    "subject.ageGroup",
    "subject.biologicalSex",
    "subject.bmi",
]).copy()

# log-transform hs-CRP
df["log_hs_crp"] = np.log1p(df["infl.hs_crp"])

formula = (
    "proportion ~ log_hs_crp + "
    "C(Q('subject.ageGroup')) + "
    "C(Q('subject.biologicalSex')) + "
    "Q('subject.bmi')"
)

fit = smf.ols(formula=formula, data=df).fit()

term = "log_hs_crp"
primary_row = pd.DataFrame([{
    "outcome": state,
    "predictor": "infl.hs_crp",
    "n_samples": int(df["sample.sampleKitGuid"].nunique()),
    "beta": fit.params.get(term, np.nan),
    "std_err": fit.bse.get(term, np.nan),
    "t_value": fit.tvalues.get(term, np.nan),
    "p_value": fit.pvalues.get(term, np.nan),
    "conf_low": fit.conf_int().loc[term, 0] if term in fit.params.index else np.nan,
    "conf_high": fit.conf_int().loc[term, 1] if term in fit.params.index else np.nan,
    "r_squared": fit.rsquared,
    "adj_r_squared": fit.rsquared_adj,
}])

primary_row.to_csv(results_dir / "dc_primary_hscrp_model_summary.csv", index=False)
df.to_csv(results_dir / "dc_primary_hscrp_model_input.csv", index=False)

# hs-CRP tertiles
df["hscrp_tertile"] = pd.qcut(
    df["infl.hs_crp"],
    q=3,
    labels=["Low hs-CRP", "Mid hs-CRP", "High hs-CRP"],
    duplicates="drop"
)

# scatter plot
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.scatter(df["infl.hs_crp"], df["proportion"], alpha=0.8, s=28)

x = np.linspace(df["infl.hs_crp"].min(), df["infl.hs_crp"].max(), 100)
plot_df = pd.DataFrame({
    "log_hs_crp": np.log1p(x),
    "subject.ageGroup": [df["subject.ageGroup"].mode().iloc[0]] * len(x),
    "subject.biologicalSex": [df["subject.biologicalSex"].mode().iloc[0]] * len(x),
    "subject.bmi": [df["subject.bmi"].median()] * len(x),
})
yhat = fit.predict(plot_df)
ax.plot(x, yhat, linewidth=2)

pval = primary_row["p_value"].iloc[0]
beta = primary_row["beta"].iloc[0]
ax.set_xlabel("hs-CRP")
ax.set_ylabel("HLA-DRhi cDC2 proportion")
ax.set_title(f"HLA-DRhi cDC2 vs hs-CRP\nβ={beta:.5f}, p={pval:.4f}")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_dc_primary_hscrp_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_dc_primary_hscrp_scatter.pdf", bbox_inches="tight")
plt.close(fig)

# tertile boxplot
groups = []
labels = []
for label in ["Low hs-CRP", "Mid hs-CRP", "High hs-CRP"]:
    vals = df.loc[df["hscrp_tertile"] == label, "proportion"].dropna().values
    if len(vals) > 0:
        groups.append(vals)
        labels.append(label)

fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.boxplot(groups, tick_labels=labels, showfliers=False)
for i, vals in enumerate(groups, start=1):
    rng = np.random.default_rng(100 + i)
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, s=24, alpha=0.8)

ax.set_ylabel("HLA-DRhi cDC2 proportion")
ax.set_title("HLA-DRhi cDC2 across hs-CRP tertiles")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_dc_primary_hscrp_tertiles.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_dc_primary_hscrp_tertiles.pdf", bbox_inches="tight")
plt.close(fig)

with open(results_dir / "dc_primary_hscrp_model_full_summary.txt", "w") as f:
    f.write(fit.summary().as_text())

print("Done.")
print(results_dir / "dc_primary_hscrp_model_summary.csv")
print(results_dir / "dc_primary_hscrp_model_full_summary.txt")
print(figures_dir / "Figure_dc_primary_hscrp_scatter.png")
print(figures_dir / "Figure_dc_primary_hscrp_tertiles.png")
