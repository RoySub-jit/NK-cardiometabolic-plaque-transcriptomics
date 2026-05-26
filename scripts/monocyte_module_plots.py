from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

donor_file = results_dir / "monocyte_module_scores_per_donor.csv"
stats_file = results_dir / "monocyte_module_scores_stats.csv"

df = pd.read_csv(donor_file)
stats = pd.read_csv(stats_file)

AGE_ORDER = ["Young Adult", "Older Adult"]
MODULE_ORDER = [
    "heatshock_score",
    "oxidative_stress_score",
    "interferon_score",
    "nfkb_inflammatory_score",
    "proteostasis_score",
]

nice_names = {
    "heatshock_score": "Heat shock",
    "oxidative_stress_score": "Oxidative stress",
    "interferon_score": "Interferon",
    "nfkb_inflammatory_score": "NF-kB / inflammatory",
    "proteostasis_score": "Proteostasis / UPR",
}

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out

def star_label(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

stats["fdr"] = bh_fdr(stats["p_value"].values)
stats.to_csv(results_dir / "monocyte_module_scores_stats_with_fdr.csv", index=False)

plot_df = df[df["subject.ageGroup"].isin(AGE_ORDER)].copy()

def add_jitter(n, center, width=0.08, seed=42):
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-width, width, size=n)

fig, axes = plt.subplots(1, len(MODULE_ORDER), figsize=(4 * len(MODULE_ORDER), 4.8), squeeze=False)
axes = axes[0]

for i, (ax, module) in enumerate(zip(axes, MODULE_ORDER)):
    sub = plot_df[["subject.ageGroup", module]].dropna().copy()
    young = sub.loc[sub["subject.ageGroup"] == "Young Adult", module].astype(float).values
    older = sub.loc[sub["subject.ageGroup"] == "Older Adult", module].astype(float).values

    ax.boxplot(
        [young, older],
        positions=[0, 1],
        widths=0.5,
        patch_artist=False,
        showfliers=False,
    )
    ax.scatter(add_jitter(len(young), 0, seed=100 + i), young, s=20, alpha=0.8)
    ax.scatter(add_jitter(len(older), 1, seed=200 + i), older, s=20, alpha=0.8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Young", "Older"])
    ax.set_title(nice_names[module], fontsize=10)
    ax.set_ylabel("Donor mean module score")

    row = stats.loc[stats["module"] == module]
    if not row.empty:
        fdr = row["fdr"].iloc[0]
        delta = row["delta_older_minus_young"].iloc[0]
        ymax = max(np.max(young) if len(young) else 0, np.max(older) if len(older) else 0)
        ymin = min(np.min(young) if len(young) else 0, np.min(older) if len(older) else 0)
        pad = max((ymax - ymin) * 0.12, 0.03)
        yline = ymax + pad
        ytext = ymax + 1.7 * pad
        ax.plot([0, 0, 1, 1], [yline, ytext, ytext, yline], lw=1)
        ax.text(0.5, ytext, f"{star_label(fdr)}\nΔ={delta:.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Monocyte donor-level stress/inflammatory module scores by age", fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "Figure_monocyte_module_boxplots.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_monocyte_module_boxplots.pdf", bbox_inches="tight")
plt.close(fig)

stats_plot = stats.copy()
stats_plot["label"] = stats_plot["module"].map(nice_names)
stats_plot = stats_plot.set_index("module").loc[MODULE_ORDER].reset_index()
stats_plot = stats_plot.sort_values("delta_older_minus_young", ascending=True)

fig, ax = plt.subplots(figsize=(7, 4.8))
y = np.arange(len(stats_plot))
x = stats_plot["delta_older_minus_young"].values

ax.axvline(0, lw=1)
ax.barh(y, x)
ax.set_yticks(y)
ax.set_yticklabels(stats_plot["label"].tolist(), fontsize=10)
ax.set_xlabel("Mean donor-level module score difference\n(Older Adult − Young Adult)")
ax.set_title("Monocyte age effect summary for stress/inflammatory modules")

xmin, xmax = np.min(x), np.max(x)
offset = max((xmax - xmin) * 0.03, 0.005)

for yi, (_, row) in enumerate(stats_plot.iterrows()):
    label = star_label(row["fdr"])
    xpos = row["delta_older_minus_young"] + (offset if row["delta_older_minus_young"] >= 0 else -offset)
    ax.text(
        xpos, yi, label,
        va="center",
        ha="left" if row["delta_older_minus_young"] >= 0 else "right",
        fontsize=9
    )

fig.tight_layout()
fig.savefig(figures_dir / "Figure_monocyte_module_effect_summary.png", dpi=300, bbox_inches="tight")
fig.savefig(figures_dir / "Figure_monocyte_module_effect_summary.pdf", bbox_inches="tight")
plt.close(fig)

print("Done.")
print(results_dir / "monocyte_module_scores_stats_with_fdr.csv")
print(figures_dir / "Figure_monocyte_module_boxplots.png")
print(figures_dir / "Figure_monocyte_module_effect_summary.png")
