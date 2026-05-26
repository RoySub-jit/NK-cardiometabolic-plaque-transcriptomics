
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
proc = base / "data/tier3_plaque_validation/GSE163154/processed"
figdir = base / "figures/tier3_plaque_validation/panel_exports"
outdir = proc
figdir.mkdir(parents=True, exist_ok=True)

expr = pd.read_csv(proc / "GSE163154_gene_level_expression.csv")
meta = pd.read_csv(proc / "GSE163154_sample_groups_extracted.csv")

# -----------------------------
# Setup
# -----------------------------
expr = expr.rename(columns={"gene_symbol": "gene"})
sample_cols = [c for c in expr.columns if c in meta["gsm"].tolist()]
mat = expr.set_index("gene")[sample_cols]

meta = meta.copy()
meta["group"] = meta["group"].astype(str)

group_order = ["No_IPH", "IPH"]
colors = {"No_IPH": "#4C78A8", "IPH": "#E45756"}

genes_main = ["GZMK", "NKG7", "GNLY", "KLRD1", "TYROBP", "FCGR3A", "PRF1", "GZMB", "CCL5", "IFITM3", "B2M"]
genes_main = [g for g in genes_main if g in mat.index]

# -----------------------------
# A. Expanded sample-level gene table + stats
# -----------------------------
sample_df = meta[["gsm", "group"]].copy()

# custom GZMK-like score
gzmk_like_genes = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1"] if g in mat.index]
if len(gzmk_like_genes) > 0:
    gzmk_score = mat.loc[gzmk_like_genes].mean(axis=0)
    sample_df["gzmk_like_score"] = sample_df["gsm"].map(gzmk_score.to_dict())

for g in genes_main:
    sample_df[g] = sample_df["gsm"].map(mat.loc[g].to_dict())

sample_df.to_csv(outdir / "GSE163154_expanded_sample_level_validation.csv", index=False)

rows = []
for feature in ["gzmk_like_score"] + genes_main:
    if feature not in sample_df.columns:
        continue
    x = pd.to_numeric(sample_df.loc[sample_df["group"] == "No_IPH", feature], errors="coerce").dropna().values
    y = pd.to_numeric(sample_df.loc[sample_df["group"] == "IPH", feature], errors="coerce").dropna().values
    if len(x) > 0 and len(y) > 0:
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        rows.append({
            "feature": feature,
            "No_IPH_mean": x.mean(),
            "IPH_mean": y.mean(),
            "No_IPH_minus_IPH": x.mean() - y.mean(),
            "mannwhitney_p": p,
            "n_No_IPH": len(x),
            "n_IPH": len(y),
        })

stats = pd.DataFrame(rows)
stats["fdr"] = multipletests(stats["mannwhitney_p"], method="fdr_bh")[1]
stats.to_csv(outdir / "GSE163154_expanded_validation_stats.csv", index=False)

# -----------------------------
# B. Bulk gene-level DE table
# -----------------------------
de_rows = []
for g in mat.index:
    x = pd.to_numeric(sample_df.loc[sample_df["group"] == "No_IPH", g], errors="coerce").dropna().values if g in sample_df.columns else mat.loc[g, meta.loc[meta["group"]=="No_IPH","gsm"]].astype(float).values
    y = pd.to_numeric(sample_df.loc[sample_df["group"] == "IPH", g], errors="coerce").dropna().values if g in sample_df.columns else mat.loc[g, meta.loc[meta["group"]=="IPH","gsm"]].astype(float).values
    if len(x) > 0 and len(y) > 0:
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        de_rows.append({
            "gene": g,
            "No_IPH_mean": x.mean(),
            "IPH_mean": y.mean(),
            "No_IPH_minus_IPH": x.mean() - y.mean(),
            "mannwhitney_p": p
        })

de = pd.DataFrame(de_rows)
de["fdr"] = multipletests(de["mannwhitney_p"], method="fdr_bh")[1]
de = de.sort_values("No_IPH_minus_IPH", ascending=False)
de.to_csv(outdir / "GSE163154_bulk_gene_level_DE.csv", index=False)

# -----------------------------
# C. Curated pathway enrichment
# -----------------------------
pathways = {
    "Cytotoxic / NK effector": ["NKG7","GNLY","PRF1","GZMB","KLRD1","FCGR3A","TYROBP","CTSW","XCL1","XCL2"],
    "Inflammatory / cytokine": ["CCL3","CCL4","CCL5","IL32","TNF","NFKBIA","JUN","FOS","LTB","IFNG","DUSP1"],
    "Interferon response": ["IFIT1","IFIT2","IFIT3","ISG15","MX1","OAS1","IFI6","IFITM1","IFITM2","IFITM3","STAT1"],
    "Migration / adhesion": ["CX3CR1","ITGAL","ITGB2","SELL","CCR7","CXCR3","CCL5","ICAM3","RAC2"],
    "Antigen presentation": ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAPBP","CD74","HLA-DRA","HLA-DRB1"],
    "Lipid / vascular relevance": ["FCGR3A","TYROBP","CX3CR1","NFKBIA","CCL5","IFITM3","JUN","FOS","NR4A1","ITGB2","B2M"]
}

universe = set(de["gene"])
sig_up = set(de.loc[(de["fdr"] < 0.05) & (de["No_IPH_minus_IPH"] > 0.15), "gene"])
sig_down = set(de.loc[(de["fdr"] < 0.05) & (de["No_IPH_minus_IPH"] < -0.15), "gene"])

enrich_rows = []
for direction, geneset in [("No_IPH > IPH", sig_up), ("IPH > No_IPH", sig_down)]:
    for pth, genes in pathways.items():
        genes = set([g for g in genes if g in universe])
        if len(genes) == 0:
            continue
        a = len(geneset & genes)
        b = len(geneset - genes)
        c = len(genes - geneset)
        d = len(universe - geneset - genes)
        odds, p = fisher_exact([[a,b],[c,d]], alternative="greater")
        enrich_rows.append({
            "direction": direction,
            "pathway": pth,
            "overlap": a,
            "set_size": len(genes),
            "odds_ratio": odds,
            "p_value": p,
            "genes_used": ", ".join(sorted(geneset & genes))
        })

enrich = pd.DataFrame(enrich_rows)
if len(enrich) > 0:
    enrich["fdr"] = enrich.groupby("direction", group_keys=False)["p_value"].transform(lambda s: multipletests(s, method="fdr_bh")[1])
    enrich["score"] = -np.log10(enrich["fdr"].clip(lower=1e-300))
enrich.to_csv(outdir / "GSE163154_curated_pathway_enrichment.csv", index=False)

# -----------------------------
# D. Sample heatmap
# -----------------------------
heat_genes = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1", "TYROBP", "FCGR3A", "PRF1", "GZMB", "CCL5", "IFITM3", "B2M"] if g in mat.index]
heat_meta = meta[["gsm","group"]].copy()
heat_meta["group_order"] = heat_meta["group"].map({"No_IPH":0, "IPH":1})
heat_meta = heat_meta.sort_values(["group_order","gsm"]).reset_index(drop=True)
heat_mat = mat.loc[heat_genes, heat_meta["gsm"]].T.copy()
heat_z = heat_mat.apply(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else x*0, axis=0)
heat_z = heat_z.replace([np.inf,-np.inf], np.nan).fillna(0)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(7.4, max(4.5, 0.22 * len(heat_meta))))
im = ax.imshow(heat_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
ax.set_title("GSE163154 sample heatmap", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(heat_genes)))
ax.set_xticklabels(heat_genes, rotation=40, ha="right")
ax.set_yticks(np.arange(len(heat_meta)))
ax.set_yticklabels(heat_meta["gsm"].tolist(), fontsize=7)

# group side bar text
for i, grp in enumerate(heat_meta["group"].tolist()):
    ax.text(-0.65, i, grp, ha="right", va="center", fontsize=7, color=colors.get(grp, "black"))

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Relative expression")
fig.tight_layout()
fig.savefig(figdir / "Panel_GSE163154_sample_heatmap.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "Panel_GSE163154_sample_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# E. Module score heatmap
# -----------------------------
module_scores = pd.DataFrame({"gsm": sample_df["gsm"], "group": sample_df["group"]})
for pth, genes in pathways.items():
    genes_use = [g for g in genes if g in mat.index]
    if len(genes_use) >= 2:
        score = mat.loc[genes_use].mean(axis=0)
        module_scores[pth] = module_scores["gsm"].map(score.to_dict())

module_scores.to_csv(outdir / "GSE163154_module_scores_by_sample.csv", index=False)

mod = module_scores.copy()
mod["group_order"] = mod["group"].map({"No_IPH":0, "IPH":1})
mod = mod.sort_values(["group_order","gsm"]).reset_index(drop=True)
module_cols = [c for c in mod.columns if c not in ["gsm","group","group_order"]]
mod_mat = mod[module_cols].copy()
mod_z = mod_mat.apply(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else x*0, axis=0)
mod_z = mod_z.replace([np.inf,-np.inf], np.nan).fillna(0)

fig, ax = plt.subplots(figsize=(6.8, max(4.0, 0.22 * len(mod))))
im = ax.imshow(mod_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
ax.set_title("GSE163154 module score heatmap", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(len(module_cols)))
ax.set_xticklabels(module_cols, rotation=40, ha="right")
ax.set_yticks(np.arange(len(mod)))
ax.set_yticklabels(mod["gsm"].tolist(), fontsize=7)

for i, grp in enumerate(mod["group"].tolist()):
    ax.text(-0.65, i, grp, ha="right", va="center", fontsize=7, color=colors.get(grp, "black"))

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Relative pathway score")
fig.tight_layout()
fig.savefig(figdir / "Panel_GSE163154_module_score_heatmap.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "Panel_GSE163154_module_score_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Professional enrichment bubble plot
# -----------------------------
if len(enrich) > 0:
    plot_df = enrich[enrich["overlap"] > 0].copy()
    plot_df = (
        plot_df.sort_values(["direction","score"], ascending=[True, False])
        .groupby("direction", observed=True)
        .head(5)
        .copy()
    )
    path_order = []
    for direction in ["No_IPH > IPH", "IPH > No_IPH"]:
        tmp = plot_df[plot_df["direction"] == direction].sort_values("score", ascending=True)
        path_order.extend(tmp["pathway"].tolist())
    seen = set()
    path_order = [x for x in path_order if not (x in seen or seen.add(x))]
    plot_df["pathway"] = pd.Categorical(plot_df["pathway"], categories=path_order, ordered=True)
    plot_df["direction"] = pd.Categorical(plot_df["direction"], categories=["No_IPH > IPH", "IPH > No_IPH"], ordered=True)
    x_map = {"No_IPH > IPH": 0, "IPH > No_IPH": 1}
    y_map = {p:i for i,p in enumerate(path_order)}
    plot_df["x"] = plot_df["direction"].map(x_map)
    plot_df["y"] = plot_df["pathway"].map(y_map)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sc = ax.scatter(
        plot_df["x"], plot_df["y"],
        s=70 + 60 * plot_df["overlap"].values,
        c=plot_df["score"].values,
        cmap="Blues",
        edgecolors="black",
        linewidths=0.5
    )
    ax.set_title("GSE163154 pathway enrichment", fontsize=12, fontweight="bold")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["No IPH > IPH", "IPH > No IPH"])
    ax.set_yticks(range(len(path_order)))
    ax.set_yticklabels(path_order)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for _, row in plot_df.iterrows():
        ax.text(row["x"] + 0.05, row["y"], f"{int(row['overlap'])}/{int(row['set_size'])}", va="center", fontsize=8)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("-log10 FDR")
    fig.tight_layout()
    fig.savefig(figdir / "Panel_GSE163154_pathway_enrichment_bubble.png", dpi=400, bbox_inches="tight")
    fig.savefig(figdir / "Panel_GSE163154_pathway_enrichment_bubble.pdf", bbox_inches="tight")
    plt.close(fig)

print("Saved main outputs to:", figdir)
print("\nExpanded validation stats:")
print(stats.sort_values("mannwhitney_p").to_string(index=False))
print("\nTop DE genes:")
print(de.sort_values("mannwhitney_p").head(20).to_string(index=False))
print("\nPathway enrichment:")
print(enrich.sort_values(['direction','score'], ascending=[True, False]).to_string(index=False))
