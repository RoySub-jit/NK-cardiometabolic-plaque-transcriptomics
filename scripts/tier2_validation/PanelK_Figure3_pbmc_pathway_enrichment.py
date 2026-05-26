
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE198339_merged_singlecell_umap.h5ad").copy()
adata.obs["leiden_r06"] = adata.obs["leiden_r06"].astype(str)

pathways = {
    "Cytotoxic / NK effector": ["NKG7","GNLY","PRF1","GZMB","CTSW","KLRD1","FCGR3A","TYROBP","TRAC","XCL1","XCL2"],
    "Inflammatory / cytokine": ["CCL3","CCL4","CCL5","IL32","TNF","NFKBIA","JUN","FOS","DUSP1","LTB","IFNG"],
    "Interferon response": ["IFIT1","IFIT2","IFIT3","ISG15","MX1","OAS1","IFI6","IFITM1","IFITM2","IFITM3","STAT1"],
    "Migration / adhesion": ["CX3CR1","ITGAL","ITGB2","SELL","CCR7","CXCR3","CCL5","ICAM3","FYN","RAC2"],
    "Antigen presentation": ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAPBP","CD74","HLA-DRA","HLA-DRB1"],
    "Lipid / vascular relevance": ["FCGR3A","TYROBP","CX3CR1","NFKBIA","CCL5","LTB","IFITM3","JUN","FOS","NR4A1","ITGB2"]
}

# keep only genes present in object
present = set(adata.var_names)
pathways = {k: [g for g in v if g in present] for k, v in pathways.items()}

# DE: cluster 1 vs 4 and cluster 4 vs 1
sub = adata[adata.obs["leiden_r06"].isin(["1","4"])].copy()
sc.tl.rank_genes_groups(sub, groupby="leiden_r06", groups=["1"], reference="4", method="wilcoxon")
de1 = sc.get.rank_genes_groups_df(sub, group="1")
sc.tl.rank_genes_groups(sub, groupby="leiden_r06", groups=["4"], reference="1", method="wilcoxon")
de4 = sc.get.rank_genes_groups_df(sub, group="4")

de1.to_csv(outdir / "GSE198339_cluster1_vs_4_DE.csv", index=False)
de4.to_csv(outdir / "GSE198339_cluster4_vs_1_DE.csv", index=False)

def enrich_direction(df, direction_name):
    d = df.copy()
    d["pvals_adj"] = pd.to_numeric(d["pvals_adj"], errors="coerce")
    d["logfoldchanges"] = pd.to_numeric(d["logfoldchanges"], errors="coerce")
    sig = d[(d["pvals_adj"] < 0.05) & (d["logfoldchanges"] > 0.25)].copy()
    sig_genes = set(sig["names"].astype(str))
    universe = set(d["names"].astype(str))
    rows = []
    for pathway, genes in pathways.items():
        genes = set(genes) & universe
        if len(genes) == 0:
            continue
        a = len(sig_genes & genes)
        b = len(sig_genes - genes)
        c = len(genes - sig_genes)
        d_ = len(universe - sig_genes - genes)
        odds, p = fisher_exact([[a,b],[c,d_]], alternative="greater")
        rows.append({
            "direction": direction_name,
            "pathway": pathway,
            "overlap": a,
            "set_size": len(genes),
            "odds_ratio": odds,
            "p_value": p,
            "genes_used": ", ".join(sorted(sig_genes & genes))
        })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out["score"] = -np.log10(out["fdr"].clip(lower=1e-300))
    return out

en1 = enrich_direction(de1, "Cluster 1 > Cluster 4")
en4 = enrich_direction(de4, "Cluster 4 > Cluster 1")
enrich = pd.concat([en1, en4], ignore_index=True)
enrich.to_csv(outdir / "GSE198339_cluster1_4_custom_pathway_enrichment.csv", index=False)

# plotting
plot_df = enrich.copy()
plot_df = plot_df.sort_values(["direction","score"], ascending=[True, False])

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), sharex=True)

for ax, direction in zip(axes, ["Cluster 1 > Cluster 4", "Cluster 4 > Cluster 1"]):
    tmp = plot_df[plot_df["direction"] == direction].copy()
    tmp = tmp.sort_values("score", ascending=True)
    y = np.arange(len(tmp))
    ax.scatter(tmp["score"], y, s=70, edgecolors="black", linewidths=0.4)
    for i, row in enumerate(tmp.itertuples()):
        ax.plot([0, row.score], [i, i], linewidth=1.2)
        ax.text(row.score + 0.03, i, f"{row.overlap}/{row.set_size}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(tmp["pathway"].tolist())
    ax.set_xlabel("-log10 FDR")
    ax.set_title(direction, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("K. PBMC mechanistic pathway enrichment: cluster 1 vs cluster 4", y=0.99, fontsize=14, fontweight="bold")
fig.tight_layout()
png = figdir / "PanelK_Figure3_pbmc_pathway_enrichment.png"
pdf = figdir / "PanelK_Figure3_pbmc_pathway_enrichment.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(enrich.to_string(index=False))
