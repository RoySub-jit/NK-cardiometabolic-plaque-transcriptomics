from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_dc_apc_restricted_sample_summary.csv")
stats_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_dc_exact_permutation_bootstrap_results.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(sample_file)
stats = pd.read_csv(stats_file)

# sort nicely
df["status_order"] = df["status"].map({"Asymptomatic": 0, "Symptomatic": 1})
df = df.sort_values(["status_order", "prop_cdc2_high"], ascending=[True, False]).reset_index(drop=True)

status_colors = {"Asymptomatic": "#4C78A8", "Symptomatic": "#E45756"}
colors = [status_colors[s] for s in df["status"]]

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

panels = [
    ("prop_cdc2_high", "Proportion of cDC2-high APC cells", "A"),
    ("cdc2_score_mean", "cDC2-like score in APC-enriched cells", "B"),
]

for ax, (col, ylabel, letter) in zip(axes, panels):
    x = np.arange(len(df))
    y = pd.to_numeric(df[col], errors="coerce").values

    ax.scatter(x, y, s=70, c=colors, edgecolors="black", linewidths=0.7, zorder=3)
    ax.plot(x, y, linewidth=1.0, alpha=0.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    row = stats[(stats["analysis"] == "full_samples") & (stats["endpoint"] == col)].iloc[0]
    ax.text(
        0.03, 0.97,
        f"Exact P = {row['exact_permutation_p']:.4f}\nΔ mean = {row['asym_minus_symptomatic_mean']:.3f}\n95% CI [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=9
    )

axes[0].scatter([], [], s=60, c=status_colors["Asymptomatic"], edgecolors="black", linewidths=0.7, label="Asymptomatic")
axes[0].scatter([], [], s=60, c=status_colors["Symptomatic"], edgecolors="black", linewidths=0.7, label="Symptomatic")
axes[0].legend(frameon=False, loc="best")

fig.suptitle("Strongest APC-restricted dendritic-cell translation signals in GSE224273", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()

png = figdir / "Figure_GSE224273_DC_strongest_signals.png"
pdf = figdir / "Figure_GSE224273_DC_strongest_signals.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
