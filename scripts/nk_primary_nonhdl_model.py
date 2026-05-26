from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

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

# primary model
formula = (
    "proportion ~ Q('lip.cholesterol_non_hdl') + "
    "C(Q('subject.ageGroup')) + "
    "C(Q('subject.biologicalSex')) + "
    "C(Q('subject.cmv')) + "
    "Q('subject.bmi')"
)

fit = smf.ols(formula=formula, data=df).fit()

term = "Q('lip.cholesterol_non_hdl')"
primary_row = pd.DataFrame([{
    "outcome": state,
    "predictor": "lip.cholesterol_non_hdl",
    "n_samples": int(df['sample.sampleKitGuid'].nunique()),
    "beta": fit.params.get(term, np.nan),
    "std_err": fit.bse.get(term, np.nan),
    "t_value": fit.tvalues.get(term, np.nan),
    "p_value": fit.pvalues.get(term, np.nan),
    "conf_low": fit.conf_int().loc[term, 0] if term in fit.params.index else np.nan,
    "conf_high": fit.conf_int().loc[term, 1] if term in fit.params.index else np.nan,
    "r_squared": fit.rsquared,
    "adj_r_squared": fit.rsquared_adj,
}])

primary_row.to_csv(results_dir / "nk_primary_nonhdl_model_summary.csv", index=False)
df.to_csv(results_dir / "nk_primary_nonhdl_model_input.csv", index=False)

# tertiles for visualization
df["nonhdl_tertile"] = pd.qcut(
    df["lip.cholesterol_non_hdl"],
    q=3,
    labels=["Low non-HDL", "Mid non-HDL", "High non-HDL"],
    duplicates="drop"
)

# scatter
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.scatter(df["lip.cholesterol_non_hdl"], df["proportion"], alpha=0.8, s=28)

x = np.linspace(df["lip.cholesterol_non_hdl"].min(), df["lip.cholesterol_non_hdl"].max(), 100)
plot_df = pd.DataFrame({
    "lip.cholesterol_non_hdl": x,
    "subject.ageGroup": [df["subject.ageGroup"].mode().iloc[0]] * len(x),
    "subject.biologicalSex": [df["subject.biologicalSex"].mode().iloc[0]] * len(x),
    "subject.cmv": [df["subject.cmv"].mode().iloc[0]] * len(x),
    "subject.bmi": [df["subject.bmi"].median()] * len(x),
})
yhat = fit.predict(plot_df)
ax.plot(x, yhat, linewidth=2)

pval = primary_row["p_value"].iloc[0]
beta = primary_row["beta"].iloc[0]
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("GZMK+ CD56dim NK proportion")
ax.set_title(f"GZMK+ CD56dim NK vs non-HDL\nβ={beta:.5f}, p={pval:.4f}")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_nk_primary_nonhdl_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_nk_primary_nonhdl_scatter.pdf", bbox_inches="tight")
plt.close(fig)

# tertile boxplot
groups = []
labels = []
for label in ["Low non-HDL", "Mid non-HDL", "High non-HDL"]:
    vals = df.loc[df["nonhdl_tertile"] == label, "proportion"].dropna().values
    if len(vals) > 0:
        groups.append(vals)
        labels.append(label)

fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.boxplot(groups, labels=labels, showfliers=False)
for i, vals in enumerate(groups, start=1):
    rng = np.random.default_rng(100 + i)
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, s=24, alpha=0.8)

ax.set_ylabel("GZMK+ CD56dim NK proportion")
ax.set_title("GZMK+ CD56dim NK across non-HDL tertiles")
fig.tight_layout()
fig.savefig(figures_dir / "Figure_nk_primary_nonhdl_tertiles.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_nk_primary_nonhdl_tertiles.pdf", bbox_inches="tight")
plt.close(fig)

with open(results_dir / "nk_primary_nonhdl_model_full_summary.txt", "w") as f:
    f.write(fit.summary().as_text())

print("Done.")
print(results_dir / "nk_primary_nonhdl_model_summary.csv")
print(results_dir / "nk_primary_nonhdl_model_full_summary.txt")
print(figures_dir / "Figure_nk_primary_nonhdl_scatter.png")
print(figures_dir / "Figure_nk_primary_nonhdl_tertiles.png")
