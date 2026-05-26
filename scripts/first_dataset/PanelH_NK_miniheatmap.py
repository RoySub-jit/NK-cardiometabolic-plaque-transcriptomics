from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

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
genes = ["GZMK", "NKG7", "GNLY", "KLRD1"]

adata = adata[adata.obs["AIFI_L3"].isin(states_order)].copy()
adata.obs["AIFI_L3"] = pd.Categorical(adata.obs["AIFI_L3"].astype(str), categories=states_order, ordered=True)

try:
    xmax = adata.X.max()
except Exception:
    xmax = None
if xmax is not None and xmax > 50:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

genes = [g for g in genes if g in adata.var_names]
expr = adata[:, genes].to_df()
expr["AIFI_L3"] = adata.obs["AIFI_L3"].astype(str).values

avg_rows = []
for state in states_order:
    sdf = expr[expr["AIFI_L3"] == state]
    avg_rows.append(sdf[genes].mean(axis=0).rename(state))
avg_df = pd.DataFrame(avg_rows).reindex(states_order)

avg_z = (avg_df - avg_df.mean(axis=0)) / avg_df.std(axis=0, ddof=0)
avg_z = avg_z.replace([np.inf, -np.inf], np.nan).fillna(0)

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

fig, ax = plt.subplots(figsize=(6.2, 4.8))

im = ax.imshow(avg_z.values, cmap="Blues", aspect="auto", vmin=-1.5, vmax=1.5)

ax.set_xticks(np.arange(len(genes)))
ax.set_xticklabels(genes)
ax.set_yticks(np.arange(len(states_order)))
ax.set_yticklabels(states_order)
ax.set_title("H", loc="left", fontweight="bold", fontsize=18)

target_row = states_order.index("GZMK+ CD56dim NK cell")
ax.add_patch(plt.Rectangle((-0.5, target_row - 0.5), len(genes), 1, fill=False, edgecolor="black", linewidth=2))

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Relative mean expression")

png = figdir / "PanelH_NK_miniheatmap.png"
pdf = figdir / "PanelH_NK_miniheatmap.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
