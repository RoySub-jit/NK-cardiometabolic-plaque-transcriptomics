
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
mat = df.pivot(index="feature", columns="dataset", values="effect").reindex(order)
piv_p = df.pivot(index="feature", columns="dataset", values="p_value").reindex(order)

mat_z = mat.copy()
for c in mat_z.columns:
    vals = mat_z[c].values.astype(float)
    sd = np.nanstd(vals)
    if sd == 0 or np.isnan(sd):
        mat_z[c] = 0
    else:
        mat_z[c] = (vals - np.nanmean(vals)) / sd

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(5.3, 5.9))
im = ax.imshow(mat_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)

ax.set_title("Cross-plaque concordance", fontsize=11.5, fontweight="bold", pad=8)
ax.set_xticks(np.arange(len(mat_z.columns)))
ax.set_xticklabels(mat_z.columns.tolist())
ax.set_yticks(np.arange(len(mat_z.index)))
ax.set_yticklabels(mat_z.index.tolist())

for i, feat in enumerate(mat.index):
    for j, ds in enumerate(mat.columns):
        val = mat.loc[feat, ds]
        p = piv_p.loc[feat, ds]
        if pd.isna(val):
            txt = "NA"
        else:
            star = "*" if pd.notna(p) and p < 0.05 else ""
            txt = f"{val:.2f}{star}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.6)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("Standardized effect direction")

png = figdir / "Panel_cross_plaque_concordance_v2.png"
pdf = figdir / "Panel_cross_plaque_concordance_v2.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("* = P < 0.05")
