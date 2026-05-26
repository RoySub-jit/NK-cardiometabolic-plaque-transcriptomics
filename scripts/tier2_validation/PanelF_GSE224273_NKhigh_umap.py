from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(base / "results/tier2_validation/GSE224273_merged_rna_raw.h5ad").copy()

# normalize/log for scoring
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]
nk_seed_genes = [g for g in nk_seed_genes if g in adata.var_names]

sc.tl.score_genes(adata, nk_seed_genes, score_name="nk_seed_score", use_raw=False)
thresh = adata.obs["nk_seed_score"].quantile(0.90)
adata.obs["is_nk_high"] = adata.obs["nk_seed_score"] >= thresh

# compute UMAP only if not present
if "X_umap" not in adata.obsm.keys():
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

umap = adata.obsm["X_umap"]
mask = adata.obs["is_nk_high"].values

plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(6.6, 5.8))

# background
ax.scatter(
    umap[~mask, 0], umap[~mask, 1],
    s=4, color="#d0d0d0", alpha=0.45, linewidths=0
)

# NK-high cells
ax.scatter(
    umap[mask, 0], umap[mask, 1],
    s=7, color="#1f78b4", alpha=0.9, linewidths=0
)

# label centroid
if mask.sum() > 0:
    cx = np.median(umap[mask, 0])
    cy = np.median(umap[mask, 1])
    ax.text(
        cx + 0.3, cy + 0.2, "NK-high cells",
        fontsize=11, fontweight="bold", color="#0b3d91",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9)
    )

ax.set_title("F", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

png = figdir / "PanelF_GSE224273_NKhigh_umap.png"
pdf = figdir / "PanelF_GSE224273_NKhigh_umap.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("NK-high threshold:", thresh)
print("Number NK-high cells:", int(mask.sum()))
