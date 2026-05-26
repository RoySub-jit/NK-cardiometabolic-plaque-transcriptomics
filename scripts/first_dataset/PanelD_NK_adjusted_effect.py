from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "results/first_dataset/nk_primary_nonhdl_model_input.csv").copy()

# basic cleanup
for col in ["proportion", "lip.cholesterol_non_hdl"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# guess available covariates from your model input
candidate_covars = [
    "demographics.age",
    "demographics.sex",
    "infectious_diseases.cmv",
    "body_measurements.body_mass_index",
    "cmv_status",
    "sex",
    "age",
    "bmi",
]
covars = [c for c in candidate_covars if c in df.columns]

keep = ["proportion", "lip.cholesterol_non_hdl"] + covars
df = df[keep].dropna().copy()

# make safe column names for formula
rename_map = {c: c.replace(".", "_").replace("-", "_") for c in df.columns}
df = df.rename(columns=rename_map)

y = "proportion"
x = "lip_cholesterol_non_hdl"
covars2 = [rename_map[c] for c in covars]

# categorical handling
for c in covars2:
    if df[c].dtype == object:
        df[c] = df[c].astype("category")

formula = y + " ~ " + x
if covars2:
    formula += " + " + " + ".join([f"C({c})" if str(df[c].dtype) == "category" else c for c in covars2])

model = smf.ols(formula, data=df).fit()

# partial residual for x
partial_resid = model.resid + model.params[x] * df[x]

# fitted line holding covariates at reference/mean
xgrid = np.linspace(df[x].min(), df[x].max(), 200)
pred_df = pd.DataFrame({x: xgrid})
for c in covars2:
    if str(df[c].dtype) == "category":
        pred_df[c] = df[c].mode().iloc[0]
    else:
        pred_df[c] = df[c].mean()

yhat = model.predict(pred_df)

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
ax.set_ylabel("Adjusted NK proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

beta = model.params[x]
pval = model.pvalues[x]
ci_low, ci_high = model.conf_int().loc[x]

ax.text(
    0.03, 0.97,
    f"Adjusted β = {beta:.6f}\nP = {pval:.4f}\n95% CI [{ci_low:.6f}, {ci_high:.6f}]",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelD_NK_adjusted_effect.png"
pdf = figdir / "PanelD_NK_adjusted_effect.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(model.summary())
