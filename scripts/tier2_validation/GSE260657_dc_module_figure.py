from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

infile = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE260657_dc_module_scores_sample_level.csv")
statsfile = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE260657_dc_module_group_stats.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(infile)
stats = pd.read_csv(statsfile)

panels = [
    ("cdc2_score_mean", "cDC2-like antigen-presentation score"),
    ("pdc_score_mean", "pDC-like score"),
    ("myeloid_score_mean", "Myeloid inflammatory score"),
    ("pdc_minus_cdc2_mean", "pDC minus cDC2 score"),
]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
axes = axes.flatten()

for ax, (col, label), panel_letter in zip(axes, panels, ["A", "B", "C", "D"]):
    a = pd.to_numeric(df.loc[df["status"] == "Asymptomatic", col], errors="coerce").dropna().values
    s = pd.to_numeric(df.loc[df["status"] == "Symptomatic", col], errors="coerce").dropna().values

    bp = ax.boxplot([a, s], tick_labels=["Asymptomatic", "Symptomatic"], widths=0.55, showfliers=False)

    rng = np.random.default_rng(42)
    ax.scatter(np.full(len(a), 1) + rng.uniform(-0.08, 0.08, len(a)), a, s=28, alpha=0.9)
    ax.scatter(np.full(len(s), 2) + rng.uniform(-0.08, 0.08, len(s)), s, s=28, alpha=0.9)

    p = float(stats.loc[stats["outcome"] == col, "mannwhitney_p"].iloc[0])
    ax.text(0.04, 0.96, f"P = {p:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ax.set_title(f"{panel_letter}  {label}", loc="left", fontweight="bold")
    ax.set_ylabel("Sample-level mean score")

fig.suptitle("Plaque-level translation of dendritic-cell and myeloid programs in GSE260657", fontsize=14, fontweight="bold")
fig.tight_layout()

png = figdir / "Figure_tier3_DC_plaque_translation.png"
pdf = figdir / "Figure_tier3_DC_plaque_translation.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
