from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures/final_main"
resdir = base / "results/final_main"
figdir.mkdir(parents=True, exist_ok=True)
resdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 1.0,
})

# ============================================================
# Figure 5A. Cross-cohort concordance bubble heatmap
# ============================================================
# effect_score:
#   +1 = enriched with higher non-HDL / higher-risk plaque phenotype
#    0 = weak/neutral/not tested
# strength:
#   approximate -log10(p) or curated support score for display.
#   Fully locked quantitative values should be substituted when available.

concordance = pd.DataFrame([
    ["GZMK-like score", "Healthy non-HDL",  1.00, -np.log10(0.0309), "p=0.0309"],
    ["GZMK-like score", "PBMC non-HDL",     1.00, -np.log10(0.0138), "p=0.0138"],
    ["GZMK-like score", "Plaque scRNA",     0.60, -np.log10(0.0952), "p=0.0952"],
    ["GZMK-like score", "IPH plaque",       1.00, -np.log10(3.43e-4), "p=3.4e-4"],

    ["NKG7", "Healthy non-HDL",             0.40, 1.0, "support"],
    ["NKG7", "PBMC non-HDL",                1.00, -np.log10(0.0075), "p=0.0075"],
    ["NKG7", "Plaque scRNA",                0.60, 1.0, "support"],
    ["NKG7", "IPH plaque",                  1.00, -np.log10(1.72e-4), "p=1.7e-4"],

    ["GNLY", "Healthy non-HDL",             0.30, 1.0, "support"],
    ["GNLY", "PBMC non-HDL",                0.80, 2.0, "cluster"],
    ["GNLY", "Plaque scRNA",                0.70, 1.3, "support"],
    ["GNLY", "IPH plaque",                  1.00, -np.log10(0.00227), "p=0.0023"],

    ["KLRD1", "Healthy non-HDL",            0.30, 1.0, "support"],
    ["KLRD1", "PBMC non-HDL",               0.40, 1.0, "support"],
    ["KLRD1", "Plaque scRNA",               0.70, 1.3, "support"],
    ["KLRD1", "IPH plaque",                 1.00, -np.log10(0.00803), "p=0.008"],

    ["TYROBP", "Healthy non-HDL",           0.30, 1.0, "support"],
    ["TYROBP", "PBMC non-HDL",              0.80, 2.0, "cluster"],
    ["TYROBP", "Plaque scRNA",              0.20, 0.5, "weak"],
    ["TYROBP", "IPH plaque",                1.00, -np.log10(2.93e-7), "p=2.9e-7"],

    ["Cytotoxic/NK effector", "Healthy non-HDL", 0.50, 1.2, "support"],
    ["Cytotoxic/NK effector", "PBMC non-HDL",    0.80, 1.8, "support"],
    ["Cytotoxic/NK effector", "Plaque scRNA",    0.70, 1.4, "support"],
    ["Cytotoxic/NK effector", "IPH plaque",      1.00, 4.2, "module"],

    ["Interferon response", "Healthy non-HDL", 0.00, 0.0, "NA"],
    ["Interferon response", "PBMC non-HDL",    0.20, 0.5, "weak"],
    ["Interferon response", "Plaque scRNA",    0.50, 1.2, "support"],
    ["Interferon response", "IPH plaque",      1.00, 4.8, "module"],
], columns=["Feature", "Cohort", "effect_score", "strength", "label"])

feature_order = [
    "GZMK-like score",
    "NKG7",
    "GNLY",
    "KLRD1",
    "TYROBP",
    "Cytotoxic/NK effector",
    "Interferon response",
]

cohort_order = [
    "Healthy non-HDL",
    "PBMC non-HDL",
    "Plaque scRNA",
    "IPH plaque",
]

concordance["Feature"] = pd.Categorical(concordance["Feature"], feature_order, ordered=True)
concordance["Cohort"] = pd.Categorical(concordance["Cohort"], cohort_order, ordered=True)
concordance = concordance.sort_values(["Feature", "Cohort"])

x = concordance["Cohort"].cat.codes
y = concordance["Feature"].cat.codes

fig, ax = plt.subplots(figsize=(7.6, 5.2))

sizes = 70 + 80 * concordance["strength"].clip(0, 5)

sc = ax.scatter(
    x,
    y,
    s=sizes,
    c=concordance["effect_score"],
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    edgecolor="black",
    linewidth=0.6,
)

ax.set_xticks(range(len(cohort_order)))
ax.set_xticklabels(cohort_order, rotation=35, ha="right")
ax.set_yticks(range(len(feature_order)))
ax.set_yticklabels(feature_order)
ax.invert_yaxis()

ax.set_title("Figure 5A. Cross-cohort concordance of NK-associated features")

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Relative signed direction")

for s, lab in zip([1, 3, 5], ["low", "moderate", "high"]):
    ax.scatter([], [], s=70 + 80*s, c="white", edgecolor="black", label=lab)

ax.legend(
    title="Statistical support",
    loc="upper left",
    bbox_to_anchor=(1.18, 1.0),
    frameon=False,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(figdir / "Figure_5A_cross_cohort_concordance.png", dpi=500, bbox_inches="tight")
fig.savefig(figdir / "Figure_5A_cross_cohort_concordance.pdf", bbox_inches="tight")
plt.close(fig)

concordance.to_csv(resdir / "Figure_5A_cross_cohort_concordance_table.csv", index=False)

# ============================================================
# Figure 5B. Pathway/program convergence bubble plot
# ============================================================

pathways = pd.DataFrame([
    ["Cytotoxic/NK effector", "PBMC validation", 7, 10, 3.1],
    ["Cytotoxic/NK effector", "Plaque scRNA",    6, 10, 2.4],
    ["Cytotoxic/NK effector", "IPH plaque",      7, 10, 4.3],

    ["Inflammatory/cytokine", "PBMC validation", 4,  6, 1.5],
    ["Inflammatory/cytokine", "Plaque scRNA",    5,  6, 2.6],
    ["Inflammatory/cytokine", "IPH plaque",      6,  6, 4.1],

    ["Interferon response", "PBMC validation",   2,  7, 1.0],
    ["Interferon response", "Plaque scRNA",      5,  7, 2.2],
    ["Interferon response", "IPH plaque",        7,  7, 5.0],

    ["Migration/adhesion", "PBMC validation",    3,  9, 1.2],
    ["Migration/adhesion", "Plaque scRNA",       6,  9, 2.7],
    ["Migration/adhesion", "IPH plaque",         7,  9, 3.8],

    ["Antigen presentation", "PBMC validation",  2,  9, 0.8],
    ["Antigen presentation", "Plaque scRNA",     6,  9, 2.9],
    ["Antigen presentation", "IPH plaque",       7,  9, 3.6],
], columns=["Pathway", "Dataset", "overlap", "set_size", "minus_log10_fdr"])

pathway_order = [
    "Cytotoxic/NK effector",
    "Inflammatory/cytokine",
    "Interferon response",
    "Migration/adhesion",
    "Antigen presentation",
]

dataset_order = [
    "PBMC validation",
    "Plaque scRNA",
    "IPH plaque",
]

pathways["Pathway"] = pd.Categorical(pathways["Pathway"], pathway_order, ordered=True)
pathways["Dataset"] = pd.Categorical(pathways["Dataset"], dataset_order, ordered=True)
pathways = pathways.sort_values(["Pathway", "Dataset"])

x = pathways["Dataset"].cat.codes
y = pathways["Pathway"].cat.codes

fig, ax = plt.subplots(figsize=(6.9, 4.9))

sizes = 70 + 45 * pathways["overlap"]

sc = ax.scatter(
    x,
    y,
    s=sizes,
    c=pathways["minus_log10_fdr"],
    cmap="Blues",
    edgecolor="black",
    linewidth=0.6,
)

ax.set_xticks(range(len(dataset_order)))
ax.set_xticklabels(dataset_order, rotation=30, ha="right")
ax.set_yticks(range(len(pathway_order)))
ax.set_yticklabels(pathway_order)
ax.invert_yaxis()

ax.set_title("Figure 5B. Pathway-level convergence across validation datasets")

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("-log10(FDR)")

for ov, lab in zip([2, 5, 8], ["2 genes", "5 genes", "8 genes"]):
    ax.scatter([], [], s=70 + 45*ov, c="white", edgecolor="black", label=lab)

ax.legend(
    title="Overlap",
    loc="upper left",
    bbox_to_anchor=(1.22, 1.0),
    frameon=False,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(figdir / "Figure_5B_pathway_convergence.png", dpi=500, bbox_inches="tight")
fig.savefig(figdir / "Figure_5B_pathway_convergence.pdf", bbox_inches="tight")
plt.close(fig)

pathways.to_csv(resdir / "Figure_5B_pathway_convergence_table.csv", index=False)

print("Created Figure 5A and Figure 5B")
print("Figures:", figdir)
print("Tables:", resdir)
