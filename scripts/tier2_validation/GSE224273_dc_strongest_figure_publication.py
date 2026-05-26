from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_dc_apc_restricted_sample_summary.csv")
stats_file = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_dc_exact_permutation_bootstrap_results.csv")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(sample_file).copy()
stats = pd.read_csv(stats_file).copy()
stats = stats[stats["analysis"] == "full_samples"].copy()

# Clean plotting order
status_order = ["Asymptomatic", "Symptomatic"]
df["status"] = pd.Categorical(df["status"], categories=status_order, ordered=True)
df = df.sort_values(["status", "sample"]).reset_index(drop=True)

colors = {
    "Asymptomatic": "#4C78A8",
    "Symptomatic": "#E45756",
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

def bootstrap_ci(values, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(samp))
    boots = np.array(boots)
    return np.quantile(boots, 0.025), np.quantile(boots, 0.975)

def draw_panel(ax, endpoint, ylabel, letter):
    panel_stats = stats[stats["endpoint"] == endpoint].iloc[0]

    x_positions = {"Asymptomatic": 0, "Symptomatic": 1}
    jitter_map = {
        "Asymptomatic": np.linspace(-0.10, 0.10, max(1, (df["status"] == "Asymptomatic").sum())),
        "Symptomatic": np.linspace(-0.07, 0.07, max(1, (df["status"] == "Symptomatic").sum())),
    }

    for status in status_order:
        sub = df[df["status"] == status].copy()
        y = pd.to_numeric(sub[endpoint], errors="coerce").values
        x0 = x_positions[status]
        jitter = jitter_map[status][:len(sub)]
        x = x0 + jitter

        # Individual points
        ax.scatter(
            x, y,
            s=70,
            color=colors[status],
            edgecolors="black",
            linewidths=0.7,
            zorder=3
        )

        # Sample labels
        for xi, yi, lab in zip(x, y, sub["sample"]):
            ax.text(xi, yi, f" {lab}", fontsize=7.5, va="center", ha="left")

        # Mean and bootstrap CI
        mean_y = np.mean(y)
        ci_low, ci_high = bootstrap_ci(y, n_boot=5000, seed=42)
        ax.errorbar(
            x0, mean_y,
            yerr=[[mean_y - ci_low], [ci_high - mean_y]],
            fmt='s',
            markersize=8,
            color='black',
            elinewidth=1.4,
            capsize=4,
            zorder=4
        )

    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Asymptomatic", "Symptomatic"])
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add one clean stats annotation
    stat_text = (
        f"Exact permutation P = {panel_stats['exact_permutation_p']:.4f}\n"
        f"Δ mean = {panel_stats['asym_minus_symptomatic_mean']:.3f}\n"
        f"Bootstrap 95% CI [{panel_stats['bootstrap_ci_low']:.3f}, {panel_stats['bootstrap_ci_high']:.3f}]"
    )
    ax.text(
        0.03, 0.97, stat_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5
    )

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

draw_panel(
    axes[0],
    endpoint="prop_cdc2_high",
    ylabel="Proportion of cDC2-high APC cells",
    letter="A"
)

draw_panel(
    axes[1],
    endpoint="cdc2_score_mean",
    ylabel="cDC2-like score in APC-enriched cells",
    letter="B"
)

fig.suptitle(
    "APC-restricted dendritic-cell translation signals in GSE224273",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

fig.tight_layout()

png = figdir / "Figure_GSE224273_DC_publication_ready.png"
pdf = figdir / "Figure_GSE224273_DC_publication_ready.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
