from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

effects_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/DC_meta_dataset_effects_all.csv")
meta_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/DC_meta_summary.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

effects = pd.read_csv(effects_file)
meta = pd.read_csv(meta_file)

endpoint_labels = {
    "cdc2_score_mean": "cDC2-like score",
    "prop_cdc2_high": "Proportion of cDC2-high APC cells",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, endpoint, letter in zip(axes, ["cdc2_score_mean", "prop_cdc2_high"], ["A", "B"]):
    sub = effects[effects["endpoint"] == endpoint].copy()
    sub = sub.sort_values("dataset").reset_index(drop=True)

    pooled = meta[
        (meta["endpoint"] == endpoint) &
        (meta["analysis"] == "primary_meta_GSE224273_GSE260657")
    ]

    y = list(range(len(sub), 0, -1))

    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color="black", linewidth=1.3)
        ax.scatter(row["hedges_g"], yi, s=45, color="black", zorder=3)
        ax.text(-2.85, yi, row["dataset"], va="center", ha="left", fontsize=10)
        ax.text(2.35, yi, f"{row['hedges_g']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]",
                va="center", ha="right", fontsize=9)

    if not pooled.empty:
        r = pooled.iloc[0]
        yp = 0.2
        ax.plot([r["ci_low"], r["ci_high"]], [yp, yp], color="black", linewidth=2.0)
        ax.scatter(r["pooled_effect"], yp, s=70, marker="D", color="black", zorder=3)
        ax.text(-2.85, yp, "Primary pooled effect", va="center", ha="left", fontsize=10, fontweight="bold")
        ax.text(2.35, yp, f"{r['pooled_effect']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
                va="center", ha="right", fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylim(-0.4, len(sub) + 1)
    ax.set_xlim(-2.9, 2.4)
    ax.set_yticks([])
    ax.set_xlabel("Hedges g (asymptomatic minus symptomatic)")
    ax.set_title(f"{letter}  {endpoint_labels[endpoint]}", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Cross-dataset dendritic-cell translational support across plaque cohorts",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

fig.tight_layout()

png = figdir / "Figure_DC_forest_supplement_clean.png"
pdf = figdir / "Figure_DC_forest_supplement_clean.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
