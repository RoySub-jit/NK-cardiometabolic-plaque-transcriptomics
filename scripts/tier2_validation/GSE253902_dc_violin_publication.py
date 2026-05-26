from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE253902_dc_apc_restricted_sample_summary.csv")
ref_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE253902_dc_apc_restricted_asym_vs_sym_reference.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(sample_file).copy()
ref = pd.read_csv(ref_file).copy()

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

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
rng = np.random.default_rng(42)

panels = [
    ("prop_cdc2_high", "Proportion of cDC2-high APC cells", "A"),
    ("cdc2_score_mean", "cDC2-like score in APC-enriched cells", "B"),
]

for ax, (endpoint, ylabel, letter) in zip(axes, panels):
    asym = pd.to_numeric(df.loc[df["status"] == "Asymptomatic", endpoint], errors="coerce").dropna().values
    sym = pd.to_numeric(df.loc[df["status"] == "Symptomatic", endpoint], errors="coerce").dropna().values

    # violin only for symptomatic, because asym has n=1
    vp = ax.violinplot(
        [sym],
        positions=[2],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body in vp["bodies"]:
        body.set_facecolor(colors["Symptomatic"])
        body.set_edgecolor("black")
        body.set_alpha(0.25)
        body.set_linewidth(1.0)

    # asymptomatic single point
    ax.scatter(
        1, asym[0],
        s=72,
        color=colors["Asymptomatic"],
        edgecolors="black",
        linewidths=0.8,
        zorder=4
    )

    # symptomatic jittered points
    x = 2 + rng.uniform(-0.07, 0.07, size=len(sym))
    ax.scatter(
        x, sym,
        s=58,
        color=colors["Symptomatic"],
        edgecolors="black",
        linewidths=0.7,
        zorder=3
    )
    ax.scatter([2], [np.mean(sym)], s=85, marker="D", color="black", zorder=4)

    row = ref[ref["outcome"] == endpoint].iloc[0]
    stat_text = (
        f"Asym − Sym mean = {row['asym_minus_sym_mean']:.3f}\n"
        f"z vs Sym = {row['z_vs_symptomatic_distribution']:.2f}\n"
        f"Percentile = {row['percentile_vs_symptomatic_distribution']:.2f}"
    )

    ax.text(
        0.97, 0.97, stat_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(status_order)
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "APC-restricted dendritic-cell reference mapping in GSE253902",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

fig.tight_layout()

png = figdir / "Figure_GSE253902_DC_violin_publication.png"
pdf = figdir / "Figure_GSE253902_DC_violin_publication.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
