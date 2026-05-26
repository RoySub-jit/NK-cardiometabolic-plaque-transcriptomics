from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv").copy()
df = df[df["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()

df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
df = df.dropna(subset=["lip.cholesterol_non_hdl", "proportion"]).copy()

# Tertiles
df["nonhdl_tertile"] = pd.qcut(df["lip.cholesterol_non_hdl"], q=3, labels=["Low", "Mid", "High"])

groups = {
    "Low": df.loc[df["nonhdl_tertile"] == "Low", "proportion"].values,
    "Mid": df.loc[df["nonhdl_tertile"] == "Mid", "proportion"].values,
    "High": df.loc[df["nonhdl_tertile"] == "High", "proportion"].values,
}

# Pairwise tests
pairs = [("Low", "Mid"), ("Mid", "High"), ("Low", "High")]
pvals = {}
for a, b in pairs:
    pvals[(a, b)] = mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue

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

fig, ax = plt.subplots(figsize=(6.6, 5.8))

positions = [1, 2, 3]
colors = ["#dbe9f6", "#8fbfe0", "#2f6fb0"]

bp = ax.boxplot(
    [groups["Low"], groups["Mid"], groups["High"]],
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.3),
    boxprops=dict(linewidth=1.0),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
)

for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_edgecolor("black")
    patch.set_alpha(0.35)

rng = np.random.default_rng(42)
for i, key in enumerate(["Low", "Mid", "High"], start=1):
    vals = groups[key]
    x = i + rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(
        x, vals,
        s=34,
        color=colors[i-1],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

ax.set_xticks(positions)
ax.set_xticklabels(["Low", "Mid", "High"])
ax.set_xlabel("Non-HDL tertile")
ax.set_ylabel("GZMK+ CD56dim NK proportion")
ax.set_title("E", loc="left", fontweight="bold", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Pairwise brackets
ymax = max([v.max() for v in groups.values()])
ymin = min([v.min() for v in groups.values()])
yr = ymax - ymin
base_y = ymax + 0.08 * yr
step = 0.09 * yr

def add_bracket(ax, x1, x2, y, text):
    h = 0.02 * yr
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + h + 0.01 * yr, text, ha="center", va="bottom", fontsize=9)

add_bracket(ax, 1, 2, base_y, f"P = {pvals[('Low','Mid')]:.3f}")
add_bracket(ax, 2, 3, base_y + step, f"P = {pvals[('Mid','High')]:.3f}")
add_bracket(ax, 1, 3, base_y + 2*step, f"P = {pvals[('Low','High')]:.3f}")

ax.set_ylim(bottom=min(0, ymin - 0.05*yr), top=base_y + 3.2*step)

png = figdir / "PanelE_NK_nonHDL_tertiles.png"
pdf = figdir / "PanelE_NK_nonHDL_tertiles.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Pairwise p-values:")
for k, v in pvals.items():
    print(k, v)

print("Saved:", png)
print("Saved:", pdf)
