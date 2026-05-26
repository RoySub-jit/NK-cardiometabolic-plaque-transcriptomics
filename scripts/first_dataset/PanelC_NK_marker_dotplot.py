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

genes = ["GZMK", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "FCGR3A", "TYROBP"]

adata = adata[adata.obs["AIFI_L3"].isin(states_order)].copy()
adata.obs["AIFI_L3"] = pd.Categorical(
    adata.obs["AIFI_L3"].astype(str),
    categories=states_order,
    ordered=True
)

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
pct_rows = []
for state in states_order:
    sdf = expr[expr["AIFI_L3"] == state]
    avg_rows.append(sdf[genes].mean(axis=0).rename(state))
    pct_rows.append((sdf[genes] > 0).mean(axis=0).rename(state))

avg_df = pd.DataFrame(avg_rows).reindex(states_order)
pct_df = pd.DataFrame(pct_rows).reindex(states_order)

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

fig = plt.figure(figsize=(9.2, 6.9))
gs = fig.add_gridspec(2, 1, height_ratios=[0.80, 0.20], hspace=0.02)

# main dotplot
ax = fig.add_subplot(gs[0])

xpos = np.arange(len(genes))
ypos = np.arange(len(states_order))

sca = None
for yi, state in enumerate(states_order):
    for xi, gene in enumerate(genes):
        size = 25 + 220 * pct_df.loc[state, gene]
        color = avg_z.loc[state, gene]
        sca = ax.scatter(
            xi, yi,
            s=size,
            c=[color],
            cmap="Blues",
            vmin=-1.5,
            vmax=1.5,
            edgecolors="black",
            linewidths=0.35
        )

ax.set_xticks(xpos)
ax.set_xticklabels(genes, rotation=35, ha="right")
ax.set_yticks(ypos)
ax.set_yticklabels(states_order)
ax.invert_yaxis()
ax.set_title("C", loc="left", fontweight="bold", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = plt.colorbar(sca, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Relative mean expression")

# bottom size legend band
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")

ax2.text(
    0.50, 0.78, "% expressing",
    fontsize=10, fontweight="bold",
    ha="center", va="center",
    transform=ax2.transAxes
)

legend_x = [0.30, 0.52, 0.74]
legend_sizes = [0.25, 0.50, 0.75]
legend_labels = ["25%", "50%", "75%"]

for x, s, lab in zip(legend_x, legend_sizes, legend_labels):
    ax2.scatter(
        x, 0.34,
        s=25 + 220 * s,
        color="white",
        edgecolors="black",
        linewidths=0.35,
        transform=ax2.transAxes,
        clip_on=False
    )
    ax2.text(
        x + 0.055, 0.34, lab,
        va="center", ha="left",
        fontsize=10,
        transform=ax2.transAxes
    )

png = figdir / "PanelC_NK_marker_dotplot.png"
pdf = figdir / "PanelC_NK_marker_dotplot.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
