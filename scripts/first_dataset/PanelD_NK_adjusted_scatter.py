from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv").copy()
df = df[df["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()

keep = [
    "proportion",
    "lip.cholesterol_non_hdl",
    "subject.ageAtFirstDraw",
    "subject.biologicalSex",
    "cmv.igg_serology_interpretation",
    "am.bmi",
]
df = df[keep].copy()

# numeric conversion
for c in ["proportion", "lip.cholesterol_non_hdl", "subject.ageAtFirstDraw", "am.bmi"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().copy()

# rename to formula-safe names
rename_map = {c: c.replace(".", "_").replace("-", "_") for c in df.columns}
df = df.rename(columns=rename_map)

# categorical columns
df["subject_biologicalSex"] = df["subject_biologicalSex"].astype("category")
df["cmv_igg_serology_interpretation"] = df["cmv_igg_serology_interpretation"].astype("category")

# full adjusted model
formula = (
    "proportion ~ lip_cholesterol_non_hdl + subject_ageAtFirstDraw + "
    "C(subject_biologicalSex) + C(cmv_igg_serology_interpretation) + am_bmi"
)

model = smf.ols(formula, data=df).fit()

# partial residual for non-HDL term
x = "lip_cholesterol_non_hdl"
partial_resid = model.resid + model.params[x] * df[x]

# adjusted fitted line holding covariates constant
xgrid = np.linspace(df[x].min(), df[x].max(), 200)
pred_df = pd.DataFrame({x: xgrid})
pred_df["subject_ageAtFirstDraw"] = df["subject_ageAtFirstDraw"].mean()
pred_df["am_bmi"] = df["am_bmi"].mean()
pred_df["subject_biologicalSex"] = df["subject_biologicalSex"].mode().iloc[0]
pred_df["cmv_igg_serology_interpretation"] = df["cmv_igg_serology_interpretation"].mode().iloc[0]

yhat = model.predict(pred_df)

beta = model.params[x]
pval = model.pvalues[x]
ci_low, ci_high = model.conf_int().loc[x]

plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(6.5, 5.8))

ax.scatter(
    df[x], partial_resid,
    s=38, color="#2f78b7", edgecolors="black", linewidths=0.5, alpha=0.9
)
ax.plot(xgrid, yhat, color="#1f4e79", linewidth=1.6)

ax.set_title("D", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("Adjusted GZMK+ CD56dim NK proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(
    0.03, 0.97,
    f"Adjusted β = {beta:.6f}\nP = {pval:.4f}\n95% CI [{ci_low:.6f}, {ci_high:.6f}]",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelD_NK_adjusted_scatter.png"
pdf = figdir / "PanelD_NK_adjusted_scatter.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Model formula:", formula)
print(model.summary())
print("Saved:", png)
print("Saved:", pdf)
