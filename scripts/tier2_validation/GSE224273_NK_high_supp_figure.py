from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_nk_high_sample_summary.csv")
stats_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_nk_high_targeted_stats.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(sample_file).copy()
stats = pd.read_csv(stats_file).copy()

status_order = ["Asymptomatic", "Symptomatic"]
df["status"] = pd.Categorical(df["status"], categories=status_order, ordered=True)

colors = {
    "Asymptomatic": "#4C78A8",
    "Symptomatic": "#E45756",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
rng = np.random.default_rng(42)

panels = [
    ("gzmk_like_score_mean", "GZMK-like score in NK-high cells", "A"),
    ("nk_core_score_mean", "NK core score in NK-high cells", "B"),
]

for ax, (endpoint, ylabel, letter) in zip(axes, panels):
    groups = []
    for status in status_order:
        vals = pd.to_numeric(df.loc[df["status"] == status, endpoint], errors="coerce").dropna().values
        groups.append(vals)

    bp = ax.boxplot(
        groups,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, status in zip(bp["boxes"], status_order):
        patch.set_facecolor(colors[status])
        patch.set_alpha(0.22)
        patch.set_edgecolor("black")

    for i, status in enumerate(status_order, start=1):
        vals = pd.to_numeric(df.loc[df["status"] == status, endpoint], errors="coerce").dropna().values
        x = i + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            x, vals,
            s=60,
            color=colors[status],
            edgecolors="black",
            linewidths=0.7,
            zorder=3
        )

    row = stats[stats["endpoint"] == endpoint].iloc[0]
    stat_text = (
        f"Exact P = {row['exact_permutation_p']:.4f}\n"
        f"Δ mean = {row['asym_minus_symptomatic_mean']:.3f}\n"
        f"95% CI [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"
    )

    ax.text(
        0.97, 0.97, stat_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.3,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.8",
            alpha=0.92
        )
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(status_order)
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].scatter([], [], s=55, c=colors["Asymptomatic"], edgecolors="black", linewidths=0.7, label="Asymptomatic")
axes[0].scatter([], [], s=55, c=colors["Symptomatic"], edgecolors="black", linewidths=0.7, label="Symptomatic")
axes[0].legend(frameon=False, loc="best")

fig.suptitle(
    "Exploratory plaque-side NK signals in GSE224273",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

fig.tight_layout()

png = figdir / "Figure_GSE224273_NK_high_supp_clean.png"
pdf = figdir / "Figure_GSE224273_NK_high_supp_clean.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
