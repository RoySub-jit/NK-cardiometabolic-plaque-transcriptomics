from pathlib import Path
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(base / "results/tier2_validation/GSE224273_merged_rna_raw.h5ad").copy()

# Normalize/log for scoring
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]
nk_seed_genes = [g for g in nk_seed_genes if g in adata.var_names]

sc.tl.score_genes(adata, nk_seed_genes, score_name="nk_seed_score", use_raw=False)
thresh = adata.obs["nk_seed_score"].quantile(0.90)
adata.obs["is_nk_high"] = adata.obs["nk_seed_score"] >= thresh

# Compute UMAP if missing
if "X_umap" not in adata.obsm.keys():
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

umap = adata.obsm["X_umap"]
mask = adata.obs["is_nk_high"].values

status = adata.obs["status"].astype(str).values if "status" in adata.obs.columns else np.array(["Unknown"] * adata.n_obs)

colors = {
    "Asymptomatic": "#1f78b4",
    "Symptomatic": "#e45756",
    "Unknown": "#888888"
}

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

fig, ax = plt.subplots(figsize=(6.8, 5.9))

# Background all cells
ax.scatter(
    umap[:, 0], umap[:, 1],
    s=4, color="#d3d3d3", alpha=0.30, linewidths=0, zorder=1
)

# Overlay NK-high by group
for grp in ["Asymptomatic", "Symptomatic", "Unknown"]:
    idx = np.where(mask & (status == grp))[0]
    if len(idx) == 0:
        continue
    ax.scatter(
        umap[idx, 0], umap[idx, 1],
        s=8,
        color=colors[grp],
        alpha=0.90,
        linewidths=0,
        label=grp,
        zorder=3
    )

ax.set_title("F", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="best")

# Add centroid labels if both groups exist
for grp in ["Asymptomatic", "Symptomatic"]:
    idx = np.where(mask & (status == grp))[0]
    if len(idx) > 0:
        cx = np.median(umap[idx, 0])
        cy = np.median(umap[idx, 1])
        ax.text(
            cx + 0.15, cy + 0.15, grp,
            fontsize=10, fontweight="bold", color=colors[grp],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.9)
        )

png = figdir / "PanelF_GSE224273_NKhigh_umap_by_group.png"
pdf = figdir / "PanelF_GSE224273_NKhigh_umap_by_group.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("NK-high threshold:", thresh)
print("NK-high cells:", int(mask.sum()))
for grp in ["Asymptomatic", "Symptomatic", "Unknown"]:
    n = int(np.sum(mask & (status == grp)))
    if n > 0:
        print(grp, n)
