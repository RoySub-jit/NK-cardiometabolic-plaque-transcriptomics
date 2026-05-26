from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_validation_participant_level.csv"
).copy()

xcol = "non_HDL"
ycol = "NK cells resting"

df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
df = df.dropna(subset=[xcol, ycol]).copy()

rho, p = spearmanr(df[xcol], df[ycol])
lr = linregress(df[xcol], df[ycol])
xline = np.linspace(df[xcol].min(), df[xcol].max(), 200)
yline = lr.intercept + lr.slope * xline

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

fig, ax = plt.subplots(figsize=(6.3, 5.6))

ax.scatter(
    df[xcol], df[ycol],
    s=42, color="#2f78b7", edgecolors="black", linewidths=0.5, alpha=0.9
)
ax.plot(xline, yline, color="#1f4e79", linewidth=1.5)

ax.set_title("D", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("NK resting proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(
    0.03, 0.97,
    f"Spearman ρ = {rho:.3f}\nP = {p:.4f}\nn = {len(df)}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelD_GSE198339_NKresting_scatter.png"
pdf = figdir / "PanelD_GSE198339_NKresting_scatter.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("Spearman rho:", rho)
print("P value:", p)
