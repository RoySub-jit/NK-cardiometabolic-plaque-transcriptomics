
from pathlib import Path
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(base / "results/tier2_validation/GSE198339_merged_singlecell_umap.h5ad")

NK_CLUSTERS = ["4", "1", "10"]

umap = adata.obsm["X_umap"]
clusters = adata.obs["leiden_r06"].astype(str).values

mask4 = clusters == "4"
mask1 = clusters == "1"
mask10 = clusters == "10"
mask_all = np.isin(clusters, NK_CLUSTERS)

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

fig, ax = plt.subplots(figsize=(7.0, 6.0))

# background
ax.scatter(
    umap[~mask_all, 0], umap[~mask_all, 1],
    s=4, color="#d3d3d3", alpha=0.25, linewidths=0, zorder=1
)

# highlighted clusters
ax.scatter(
    umap[mask4, 0], umap[mask4, 1],
    s=8, color="#1f78b4", alpha=0.90, linewidths=0,
    label="Cluster 4: canonical NK", zorder=3
)
ax.scatter(
    umap[mask1, 0], umap[mask1, 1],
    s=8, color="#4daf4a", alpha=0.90, linewidths=0,
    label="Cluster 1: GZMK-enriched NK-like", zorder=3
)
ax.scatter(
    umap[mask10, 0], umap[mask10, 1],
    s=10, color="#984ea3", alpha=0.95, linewidths=0,
    label="Cluster 10: minor NK-like", zorder=3
)

for cl, lab, color in [
    ("4", "4", "#1f78b4"),
    ("1", "1", "#4daf4a"),
    ("10", "10", "#984ea3"),
]:
    idx = np.where(clusters == cl)[0]
    if len(idx) > 0:
        cx = np.median(umap[idx, 0])
        cy = np.median(umap[idx, 1])
        ax.text(
            cx + 0.15, cy + 0.15, lab,
            fontsize=10, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.8", alpha=0.9)
        )

ax.set_title("G", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="best", fontsize=9)

png = figdir / "PanelG_GSE198339_NKcluster_umap.png"
pdf = figdir / "PanelG_GSE198339_NKcluster_umap.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("Cluster 4 cells:", int(mask4.sum()))
print("Cluster 1 cells:", int(mask1.sum()))
print("Cluster 10 cells:", int(mask10.sum()))
