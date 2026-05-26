from pathlib import Path
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(base / "data/first_dataset/raw/human_immune_health_atlas_nk-ilc.h5ad")

states_order = [
    "GZMK- CD56dim NK cell",
    "GZMK+ CD56dim NK cell",
    "Adaptive NK cell",
    "ISG+ CD56dim NK cell",
    "CD56bright NK cell",
    "Proliferating NK cell",
    "ILC",
]

adata = adata[adata.obs["AIFI_L3"].isin(states_order)].copy()
adata.obs["AIFI_L3"] = adata.obs["AIFI_L3"].astype(str)

umap = adata.obsm["X_umap"]

state_colors = {
    "GZMK- CD56dim NK cell": "#bcd7f0",
    "GZMK+ CD56dim NK cell": "#1f78b4",
    "Adaptive NK cell": "#fdb462",
    "ISG+ CD56dim NK cell": "#b2df8a",
    "CD56bright NK cell": "#cab2d6",
    "Proliferating NK cell": "#fb9a99",
    "ILC": "#bdbdbd",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8.8, 6.8))

# plot background states first
for state in states_order:
    idx = np.where(adata.obs["AIFI_L3"].values == state)[0]
    if len(idx) == 0:
        continue

    if state == "GZMK+ CD56dim NK cell":
        ax.scatter(
            umap[idx, 0], umap[idx, 1],
            s=4.0, c=state_colors[state],
            alpha=0.9, linewidths=0,
            label=state, zorder=3
        )
    else:
        ax.scatter(
            umap[idx, 0], umap[idx, 1],
            s=2.2, c=state_colors[state],
            alpha=0.45, linewidths=0,
            label=state, zorder=1
        )

# label the GZMK+ cluster at its centroid
gidx = np.where(adata.obs["AIFI_L3"].values == "GZMK+ CD56dim NK cell")[0]
gx = np.median(umap[gidx, 0])
gy = np.median(umap[gidx, 1])
ax.text(
    gx + 0.6, gy + 0.2,
    "GZMK+ CD56dim NK",
    fontsize=11, fontweight="bold",
    color="#0b3d91",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9)
)

ax.set_title("B", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=9,
    markerscale=2.5,
    handletextpad=0.4
)

png = figdir / "PanelB_NK_UMAP.png"
pdf = figdir / "PanelB_NK_UMAP.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
