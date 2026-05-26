
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

enrich = pd.read_csv(outdir / "GSE198339_cluster1_4_custom_pathway_enrichment.csv").copy()
de1 = pd.read_csv(outdir / "GSE198339_cluster1_vs_4_DE.csv").copy()
de4 = pd.read_csv(outdir / "GSE198339_cluster4_vs_1_DE.csv").copy()

# keep only informative pathways
enrich["score"] = pd.to_numeric(enrich["score"], errors="coerce")
enrich["overlap"] = pd.to_numeric(enrich["overlap"], errors="coerce")
enrich["fdr"] = pd.to_numeric(enrich["fdr"], errors="coerce")

enrich = enrich[(enrich["overlap"] > 0)].copy()
enrich = enrich.sort_values(["direction", "score"], ascending=[True, False])

# keep top pathways per direction
top_n = 4
plot_df = (
    enrich.groupby("direction", observed=True)
    .head(top_n)
    .copy()
)

# gene direction table
de1["names"] = de1["names"].astype(str)
de4["names"] = de4["names"].astype(str)
de1["logfoldchanges"] = pd.to_numeric(de1["logfoldchanges"], errors="coerce")
de4["logfoldchanges"] = pd.to_numeric(de4["logfoldchanges"], errors="coerce")

gene_dir = {}
for _, r in de1.iterrows():
    gene = r["names"]
    gene_dir[gene] = ("Cluster 1 > 4", r["logfoldchanges"])
for _, r in de4.iterrows():
    gene = r["names"]
    if gene not in gene_dir:
        gene_dir[gene] = ("Cluster 4 > 1", r["logfoldchanges"])

# build bipartite graph
G = nx.Graph()

path_nodes = []
gene_nodes = []

for _, row in plot_df.iterrows():
    pth = row["pathway"]
    direction = row["direction"]
    genes = [g.strip() for g in str(row["genes_used"]).split(",") if g.strip()]
    if len(genes) == 0:
        continue

    path_nodes.append(pth)
    G.add_node(
        pth,
        node_type="pathway",
        direction=direction,
        score=row["score"],
        overlap=row["overlap"]
    )

    for g in genes:
        gene_nodes.append(g)
        d = gene_dir.get(g, ("Unknown", 0.0))
        G.add_node(
            g,
            node_type="gene",
            direction=d[0],
            logfc=d[1]
        )
        G.add_edge(pth, g)

path_nodes = list(dict.fromkeys(path_nodes))
gene_nodes = list(dict.fromkeys(gene_nodes))

# positions: pathways left, genes right
pos = {}
path_y = np.linspace(1, 0, len(path_nodes)) if len(path_nodes) > 1 else np.array([0.5])
gene_y = np.linspace(1, 0, len(gene_nodes)) if len(gene_nodes) > 1 else np.array([0.5])

for y, pth in zip(path_y, path_nodes):
    pos[pth] = (0.0, y)

for y, g in zip(gene_y, gene_nodes):
    pos[g] = (1.0, y)

# styles
path_color_map = {
    "Cluster 1 > 4": "#4daf4a",
    "Cluster 4 > 1": "#1f78b4"
}
gene_color_map = {
    "Cluster 1 > 4": "#7fc97f",
    "Cluster 4 > 1": "#80b1d3",
    "Unknown": "#bdbdbd"
}

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(10.8, 6.6))

# edges
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    ax.plot([x0, x1], [y0, y1], color="0.75", linewidth=1.1, zorder=1)

# pathway nodes
for pth in path_nodes:
    x, y = pos[pth]
    node = G.nodes[pth]
    ax.scatter(
        x, y,
        s=220 + 90 * node["overlap"],
        color=path_color_map.get(node["direction"], "#cccccc"),
        edgecolors="black",
        linewidths=0.7,
        zorder=3
    )
    ax.text(
        x - 0.03, y, pth,
        ha="right", va="center",
        fontsize=9.2, fontweight="bold"
    )

# gene nodes
for g in gene_nodes:
    x, y = pos[g]
    node = G.nodes[g]
    ax.scatter(
        x, y,
        s=120,
        color=gene_color_map.get(node["direction"], "#bdbdbd"),
        edgecolors="black",
        linewidths=0.6,
        zorder=3
    )
    ax.text(
        x + 0.03, y, g,
        ha="left", va="center",
        fontsize=9
    )

# headers
ax.text(0.0, 1.06, "Enriched pathways", ha="left", va="bottom", fontsize=11, fontweight="bold")
ax.text(1.0, 1.06, "Driving genes", ha="right", va="bottom", fontsize=11, fontweight="bold")

# title
ax.set_title("K", loc="left", fontweight="bold", fontsize=18)

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#4daf4a", markeredgecolor="black", markersize=8, label="Pathway: Cluster 1 > 4"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#1f78b4", markeredgecolor="black", markersize=8, label="Pathway: Cluster 4 > 1"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#7fc97f", markeredgecolor="black", markersize=8, label="Gene: Cluster 1 > 4"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#80b1d3", markeredgecolor="black", markersize=8, label="Gene: Cluster 4 > 1"),
]
ax.legend(handles=legend_elements, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.08, 1.08)
ax.axis("off")

png = figdir / "PanelK_Figure3_pbmc_pathway_gene_network.png"
pdf = figdir / "PanelK_Figure3_pbmc_pathway_gene_network.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("Pathways used:")
print(plot_df[["direction", "pathway", "overlap", "score", "genes_used"]].to_string(index=False))
