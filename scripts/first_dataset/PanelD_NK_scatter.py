from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv").copy()
df = df[df["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()

df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
df = df.dropna(subset=["lip.cholesterol_non_hdl", "proportion"]).copy()

lr = linregress(df["lip.cholesterol_non_hdl"], df["proportion"])
xline = np.linspace(df["lip.cholesterol_non_hdl"].min(), df["lip.cholesterol_non_hdl"].max(), 200)
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

fig, ax = plt.subplots(figsize=(6.4, 5.7))

ax.scatter(
    df["lip.cholesterol_non_hdl"], df["proportion"],
    s=38, color="#2f78b7", edgecolors="black", linewidths=0.5, alpha=0.9
)
ax.plot(xline, yline, color="#1f4e79", linewidth=1.5)

ax.set_title("D", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("GZMK+ CD56dim NK proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(
    0.03, 0.97,
    f"n = {len(df)}\nP = {lr.pvalue:.4f}\nR² = {lr.rvalue**2:.3f}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelD_NK_scatter.png"
pdf = figdir / "PanelD_NK_scatter.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
