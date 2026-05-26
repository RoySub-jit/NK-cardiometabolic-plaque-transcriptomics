
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
outdir = base / "results" / "tier2_validation"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE198339_merged_singlecell_umap.h5ad").copy()
adata.obs["leiden_r06"] = adata.obs["leiden_r06"].astype(str)

clusters_use = ["1", "4"]
genes = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"] if g in adata.var_names]

sub = adata[adata.obs["leiden_r06"].isin(clusters_use)].copy()
expr = sub[:, genes].to_df()
expr["cluster"] = sub.obs["leiden_r06"].values

rows = []
for g in genes:
    x1 = pd.to_numeric(expr.loc[expr["cluster"] == "1", g], errors="coerce").dropna().values
    x4 = pd.to_numeric(expr.loc[expr["cluster"] == "4", g], errors="coerce").dropna().values
    p = mannwhitneyu(x1, x4, alternative="two-sided").pvalue
    rows.append({
        "gene": g,
        "cluster1_mean": x1.mean(),
        "cluster4_mean": x4.mean(),
        "cluster1_minus_cluster4": x1.mean() - x4.mean(),
        "mannwhitney_p": p,
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv(outdir / "GSE198339_cluster1_vs_4_gene_stats.csv", index=False)

mat = stats_df.set_index("gene")[["cluster1_mean", "cluster4_mean"]]
mat_z = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
mat_z = mat_z.replace([np.inf, -np.inf], np.nan).fillna(0)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(5.2, 6.2))
im = ax.imshow(mat_z.values, aspect="auto", cmap="Blues", vmin=-1.5, vmax=1.5)

ax.set_title("J", loc="left", fontweight="bold", fontsize=18)
ax.set_yticks(np.arange(len(mat_z.index)))
ax.set_yticklabels(mat_z.index.tolist())
ax.set_xticks([0, 1])
ax.set_xticklabels(["Cluster 1", "Cluster 4"])

# Put p-values inside the left tile row so they don't collide with the colorbar
for i, gene in enumerate(stats_df["gene"]):
    p = stats_df.loc[stats_df["gene"] == gene, "mannwhitney_p"].iloc[0]
    ptxt = f"P={p:.1e}" if p < 0.001 else f"P={p:.3f}"
    ax.text(
        0, i, ptxt,
        ha="center", va="center",
        fontsize=8.2, color="white",
        bbox=dict(boxstyle="round,pad=0.16", facecolor=(0,0,0,0.18), edgecolor="none")
    )

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label("Relative expression")

png = figdir / "PanelJ_GSE198339_cluster1_vs_4_mechanistic.png"
pdf = figdir / "PanelJ_GSE198339_cluster1_vs_4_mechanistic.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(stats_df.to_string(index=False))
