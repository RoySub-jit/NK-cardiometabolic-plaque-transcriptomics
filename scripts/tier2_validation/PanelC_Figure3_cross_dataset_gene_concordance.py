
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

pbmc = pd.read_csv(outdir / "GSE198339_cluster1_vs_4_gene_stats.csv")
plaque = pd.read_csv(outdir / "GSE224273_nk_high_core_gene_stats.csv")

genes = ["GZMK", "NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]

pbmc = pbmc[pbmc["gene"].isin(genes)].copy()
plaque = plaque[plaque["gene"].isin(genes)].copy()

# unify columns
pbmc = pbmc[["gene", "cluster1_minus_cluster4", "mannwhitney_p"]].rename(columns={
    "cluster1_minus_cluster4": "effect",
    "mannwhitney_p": "p_value"
})
pbmc["dataset"] = "PBMC\nCluster1 - Cluster4"

plaque = plaque[["gene", "asym_minus_sym_mean", "exact_permutation_p"]].rename(columns={
    "asym_minus_sym_mean": "effect",
    "exact_permutation_p": "p_value"
})
plaque["dataset"] = "Plaque\nAsym - Sym"

df = pd.concat([pbmc, plaque], ignore_index=True)

# wide matrix
mat = df.pivot(index="gene", columns="dataset", values="effect").reindex(genes)
piv_p = df.pivot(index="gene", columns="dataset", values="p_value").reindex(genes)

# z-score across columns for visual comparability
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

fig, ax = plt.subplots(figsize=(5.6, 5.8))
im = ax.imshow(mat_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)

ax.set_title("C", loc="left", fontweight="bold", fontsize=18)
ax.set_xticks(np.arange(len(mat_z.columns)))
ax.set_xticklabels(mat_z.columns.tolist())
ax.set_yticks(np.arange(len(mat_z.index)))
ax.set_yticklabels(mat_z.index.tolist())

for i, gene in enumerate(mat.index):
    for j, ds in enumerate(mat.columns):
        p = piv_p.loc[gene, ds]
        mark = "*" if pd.notna(p) and p < 0.05 else ""
        txt = f"{mat.loc[gene, ds]:.2f}{mark}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("Relative effect direction")

png = figdir / "PanelC_Figure3_cross_dataset_gene_concordance.png"
pdf = figdir / "PanelC_Figure3_cross_dataset_gene_concordance.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(mat)
print(piv_p)
