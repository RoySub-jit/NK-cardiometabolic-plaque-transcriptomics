
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

enrich = pd.read_csv(outdir / "GSE198339_cluster1_4_custom_pathway_enrichment.csv").copy()

# keep only informative rows
enrich = enrich[enrich["overlap"] > 0].copy()

# optional: keep top pathways per direction by score
enrich["score"] = pd.to_numeric(enrich["score"], errors="coerce")
enrich["overlap"] = pd.to_numeric(enrich["overlap"], errors="coerce")
enrich["set_size"] = pd.to_numeric(enrich["set_size"], errors="coerce")
enrich["fdr"] = pd.to_numeric(enrich["fdr"], errors="coerce")

top_n = 5
plot_df = (
    enrich.sort_values(["direction", "score"], ascending=[True, False])
    .groupby("direction", observed=True)
    .head(top_n)
    .copy()
)

# preserve readable order
path_order = []
for direction in ["Cluster 1 > Cluster 4", "Cluster 4 > Cluster 1"]:
    tmp = plot_df[plot_df["direction"] == direction].sort_values("score", ascending=True)
    path_order.extend(tmp["pathway"].tolist())

# unique while preserving order
seen = set()
path_order = [x for x in path_order if not (x in seen or seen.add(x))]

plot_df["pathway"] = pd.Categorical(plot_df["pathway"], categories=path_order, ordered=True)
plot_df["direction"] = pd.Categorical(
    plot_df["direction"],
    categories=["Cluster 1 > Cluster 4", "Cluster 4 > Cluster 1"],
    ordered=True
)

x_map = {"Cluster 1 > Cluster 4": 0, "Cluster 4 > Cluster 1": 1}
y_map = {p: i for i, p in enumerate(path_order)}

plot_df["x"] = plot_df["direction"].map(x_map)
plot_df["y"] = plot_df["pathway"].map(y_map)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(7.0, 5.8))

sc = ax.scatter(
    plot_df["x"],
    plot_df["y"],
    s=55 + 55 * plot_df["overlap"].values,
    c=plot_df["score"].values,
    cmap="Blues",
    vmin=0,
    vmax=max(2.0, np.nanmax(plot_df["score"].values)),
    edgecolors="black",
    linewidths=0.5
)

ax.set_title("K", loc="left", fontweight="bold", fontsize=18)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Cluster 1 > Cluster 4", "Cluster 4 > Cluster 1"])
ax.set_yticks(range(len(path_order)))
ax.set_yticklabels(path_order)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotate overlap counts inside/near bubbles
for _, row in plot_df.iterrows():
    ax.text(
        row["x"] + 0.06, row["y"],
        f"{int(row['overlap'])}/{int(row['set_size'])}",
        va="center", ha="left", fontsize=8.3
    )

cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("-log10 FDR")

# size legend
for size_val, label in [(1, "1 gene"), (3, "3 genes"), (5, "5 genes")]:
    ax.scatter([], [], s=55 + 55 * size_val, c="white", edgecolors="black", linewidths=0.5, label=label)
ax.legend(title="Overlap", frameon=False, loc="lower right")

png = figdir / "PanelK_Figure3_pbmc_pathway_enrichment_bubble.png"
pdf = figdir / "PanelK_Figure3_pbmc_pathway_enrichment_bubble.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(plot_df[["direction", "pathway", "overlap", "set_size", "score", "fdr"]].to_string(index=False))
