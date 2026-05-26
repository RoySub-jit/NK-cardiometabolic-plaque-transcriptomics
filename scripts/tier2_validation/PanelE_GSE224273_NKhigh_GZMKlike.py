from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

sample_file = base / "results/tier2_validation/GSE224273_nk_high_sample_summary.csv"
stats_file = base / "results/tier2_validation/GSE224273_nk_high_targeted_stats.csv"

df = pd.read_csv(sample_file).copy()
stats = pd.read_csv(stats_file).copy()

status_order = ["Asymptomatic", "Symptomatic"]
df["status"] = pd.Categorical(df["status"], categories=status_order, ordered=True)

row = stats[stats["endpoint"] == "gzmk_like_score_mean"].iloc[0]

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

fig = plt.figure(figsize=(6.6, 5.8))
gs = fig.add_gridspec(2, 1, height_ratios=[0.24, 0.76], hspace=0.03)

ax_top = fig.add_subplot(gs[0])
ax_top.axis("off")
ax_top.text(0.0, 0.92, "E", fontsize=18, fontweight="bold", ha="left", va="top")
ax_top.text(
    0.08, 0.62,
    f"Exact permutation P = {row['exact_permutation_p']:.4f}\n"
    f"Δ mean = {row['asym_minus_symptomatic_mean']:.3f}\n"
    f"95% CI [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]",
    fontsize=9.5, ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95)
)

ax = fig.add_subplot(gs[1])

groups = []
for s in status_order:
    vals = pd.to_numeric(df.loc[df["status"] == s, "gzmk_like_score_mean"], errors="coerce").dropna().values
    groups.append(vals)

bp = ax.boxplot(
    groups,
    positions=[1, 2],
    widths=0.52,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    boxprops=dict(linewidth=1.0),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
)

fills = {"Asymptomatic": "#4C78A8", "Symptomatic": "#E45756"}
for patch, s in zip(bp["boxes"], status_order):
    patch.set_facecolor(fills[s])
    patch.set_alpha(0.28)
    patch.set_edgecolor("black")

rng = np.random.default_rng(42)
for i, s in enumerate(status_order, start=1):
    vals = pd.to_numeric(df.loc[df["status"] == s, "gzmk_like_score_mean"], errors="coerce").dropna().values
    x = i + rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(x, vals, s=55, color=fills[s], edgecolors="black", linewidths=0.6, zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels(status_order)
ax.set_ylabel("GZMK-like score in NK-high cells")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

png = figdir / "PanelE_GSE224273_NKhigh_GZMKlike.png"
pdf = figdir / "PanelE_GSE224273_NKhigh_GZMKlike.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
