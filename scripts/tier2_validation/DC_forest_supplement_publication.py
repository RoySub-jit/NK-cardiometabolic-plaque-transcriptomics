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

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

for ax, endpoint, letter in zip(axes, ["cdc2_score_mean", "prop_cdc2_high"], ["A", "B"]):
    sub = effects[effects["endpoint"] == endpoint].copy()

    # Keep only datasets with valid effect sizes for plotting
    sub = sub.dropna(subset=["hedges_g", "ci_low", "ci_high"]).copy()
    sub = sub.sort_values("dataset").reset_index(drop=True)

    pooled = meta[
        (meta["endpoint"] == endpoint) &
        (meta["analysis"] == "primary_meta_GSE224273_GSE260657")
    ].copy()

    # Fixed row layout
    y_rows = list(range(len(sub), 0, -1))   # e.g. 2,1
    pooled_y = 0

    # Plot dataset rows
    for y, (_, row) in zip(y_rows, sub.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color="black", linewidth=1.4)
        ax.scatter(row["hedges_g"], y, s=46, color="black", zorder=3)

        ax.text(-2.70, y, row["dataset"], va="center", ha="left", fontsize=10)
        ax.text(2.45, y, f"{row['hedges_g']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]",
                va="center", ha="right", fontsize=9)

    # Plot pooled row separately
    if not pooled.empty:
        r = pooled.iloc[0]
        ax.plot([r["ci_low"], r["ci_high"]], [pooled_y, pooled_y], color="black", linewidth=2.0)
        ax.scatter(r["pooled_effect"], pooled_y, s=72, marker="D", color="black", zorder=3)

        ax.text(-2.70, pooled_y, "Primary pooled effect", va="center", ha="left",
                fontsize=10, fontweight="bold")
        ax.text(2.45, pooled_y,
                f"{r['pooled_effect']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
                va="center", ha="right", fontsize=9, fontweight="bold")

    # Separator above pooled row
    ax.hlines(0.5, -2.75, 2.5, colors="0.75", linewidth=0.8)

    # Zero line
    ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0)

    ax.set_ylim(-0.6, len(sub) + 0.8)
    ax.set_xlim(-2.8, 2.5)
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

png = figdir / "Figure_DC_forest_supplement_publication.png"
pdf = figdir / "Figure_DC_forest_supplement_publication.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
