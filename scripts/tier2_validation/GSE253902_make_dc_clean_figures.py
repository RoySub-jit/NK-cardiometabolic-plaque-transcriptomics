from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(outdir / "GSE253902_dc_presence_sample_scores_fixed.csv").copy()

# Keep only needed columns
df = df[["sample", "status", "n_cells", "cdc2_score_mean", "pdc_score_mean", "pdc_minus_cdc2_mean"]].copy()

# Sort: asymptomatic first, then symptomatic by descending cDC2-like score
df["status_order"] = df["status"].map({"Asymptomatic": 0, "Symptomatic": 1})
df = df.sort_values(["status_order", "cdc2_score_mean"], ascending=[True, False]).reset_index(drop=True)

# Consistent styling
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

status_colors = {
    "Asymptomatic": "#4C78A8",
    "Symptomatic": "#E45756",
}
colors = [status_colors[s] for s in df["status"]]

def make_clean_plot(ycol, ylabel, title, outname):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x = range(len(df))
    y = pd.to_numeric(df[ycol], errors="coerce")

    ax.scatter(x, y, s=65, c=colors, edgecolors="black", linewidths=0.7, zorder=3)

    # connect points lightly to aid reading, but keep clean
    ax.plot(x, y, linewidth=1.0, alpha=0.5, zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["sample"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # small legend-like annotation
    ax.scatter([], [], s=55, c=status_colors["Asymptomatic"], edgecolors="black", linewidths=0.7, label="Asymptomatic")
    ax.scatter([], [], s=55, c=status_colors["Symptomatic"], edgecolors="black", linewidths=0.7, label="Symptomatic")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()

    png = figdir / f"{outname}.png"
    pdf = figdir / f"{outname}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)

# Figure A
make_clean_plot(
    ycol="cdc2_score_mean",
    ylabel="Sample-level mean cDC2-like score",
    title="Plaque cDC2-like antigen-presentation program",
    outname="Figure_tier3_DC_cdc2_clean"
)

# Figure B
make_clean_plot(
    ycol="pdc_minus_cdc2_mean",
    ylabel="Sample-level mean pDC minus cDC2 score",
    title="Plaque pDC-to-cDC2 balance",
    outname="Figure_tier3_DC_pdc_minus_cdc2_clean"
)
