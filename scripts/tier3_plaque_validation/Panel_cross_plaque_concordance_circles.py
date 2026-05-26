
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures/tier3_plaque_validation/panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

# Cohort 1
c1_main = pd.read_csv(base / "results/tier2_validation/GSE224273_nk_high_targeted_stats.csv")
c1_gene = pd.read_csv(base / "results/tier2_validation/GSE224273_nk_high_core_gene_stats.csv")

# Cohort 2
c2 = pd.read_csv(base / "data/tier3_plaque_validation/GSE163154/processed/GSE163154_nk_validation_stats.csv")

rows = []

# GSE163154
rows.append({
    "feature": "GZMK-like score",
    "dataset": "GSE163154\nNo IPH vs IPH",
    "effect": float(c2.loc[c2["feature"] == "gzmk_like_score", "No_IPH_minus_IPH"].iloc[0]),
    "p_value": float(c2.loc[c2["feature"] == "gzmk_like_score", "mannwhitney_p"].iloc[0]),
})
for g in ["NKG7", "GNLY", "KLRD1", "TYROBP"]:
    if g in c2["feature"].values:
        r = c2.loc[c2["feature"] == g].iloc[0]
        rows.append({
            "feature": g,
            "dataset": "GSE163154\nNo IPH vs IPH",
            "effect": float(r["No_IPH_minus_IPH"]),
            "p_value": float(r["mannwhitney_p"]),
        })

# GSE224273
rows.append({
    "feature": "GZMK-like score",
    "dataset": "GSE224273\nAsym vs Sym",
    "effect": float(c1_main.loc[c1_main["endpoint"] == "gzmk_like_score_mean", "asym_minus_symptomatic_mean"].iloc[0]),
    "p_value": float(c1_main.loc[c1_main["endpoint"] == "gzmk_like_score_mean", "exact_permutation_p"].iloc[0]),
})
for g in ["NKG7", "GNLY", "KLRD1"]:
    if g in c1_gene["gene"].values:
        r = c1_gene.loc[c1_gene["gene"] == g].iloc[0]
        rows.append({
            "feature": g,
            "dataset": "GSE224273\nAsym vs Sym",
            "effect": float(r["asym_minus_sym_mean"]),
            "p_value": float(r["exact_permutation_p"]),
        })

df = pd.DataFrame(rows)

order = ["GZMK-like score", "NKG7", "GNLY", "KLRD1", "TYROBP"]
datasets = ["GSE163154\nNo IPH vs IPH", "GSE224273\nAsym vs Sym"]

mat = df.pivot(index="feature", columns="dataset", values="effect").reindex(order)[datasets]
piv_p = df.pivot(index="feature", columns="dataset", values="p_value").reindex(order)[datasets]

# standardize by column for color
mat_z = mat.copy()
for c in mat_z.columns:
    vals = mat_z[c].values.astype(float)
    sd = np.nanstd(vals)
    if sd == 0 or np.isnan(sd):
        mat_z[c] = 0
    else:
        mat_z[c] = (vals - np.nanmean(vals)) / sd

# size based on abs(raw effect)
absmax = np.nanmax(np.abs(mat.values))
if absmax == 0 or np.isnan(absmax):
    absmax = 1.0

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(5.6, 6.0))

x_map = {ds:i for i, ds in enumerate(datasets)}
y_map = {feat:i for i, feat in enumerate(order)}

for feat in order:
    for ds in datasets:
        x = x_map[ds]
        y = y_map[feat]
        val = mat.loc[feat, ds]
        z = mat_z.loc[feat, ds]
        p = piv_p.loc[feat, ds]

        if pd.isna(val):
            ax.text(x, y, "NA", ha="center", va="center", fontsize=8.5, color="0.4")
            continue

        size = 350 + 1450 * (abs(val) / absmax)
        sc = ax.scatter(
            x, y,
            s=size,
            c=[z],
            cmap="RdBu_r",
            vmin=-1.6, vmax=1.6,
            edgecolors="black",
            linewidths=0.6,
            zorder=3
        )

        star = "*" if pd.notna(p) and p < 0.05 else ""
        txt = f"{val:.2f}{star}"

        # choose text color based on color intensity
        txt_color = "white" if abs(z) > 0.7 else "black"
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.4, color=txt_color, zorder=4)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()

ax.set_title("Cross-plaque concordance", fontsize=11.5, fontweight="bold", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("Standardized effect direction")

# size legend
for s, lab in [(0.25, "smaller effect"), (0.6, "medium"), (1.0, "larger effect")]:
    ax.scatter([], [], s=350 + 1450*s, c="white", edgecolors="black", linewidths=0.6, label=lab)
ax.legend(title="Absolute effect", frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.3)

ax.text(
    0.0, -0.24,
    "* P < 0.05",
    transform=ax.transAxes,
    ha="left", va="top", fontsize=8.5
)

png = figdir / "Panel_cross_plaque_concordance_circles.png"
pdf = figdir / "Panel_cross_plaque_concordance_circles.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
