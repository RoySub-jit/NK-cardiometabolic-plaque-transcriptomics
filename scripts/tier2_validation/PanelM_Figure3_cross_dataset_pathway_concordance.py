
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

# ---------- PBMC ----------
pbmc = sc.read_h5ad(outdir / "GSE198339_merged_singlecell_umap.h5ad").copy()
pbmc.obs["leiden_r06"] = pbmc.obs["leiden_r06"].astype(str)

# ---------- Plaque ----------
plaque = sc.read_h5ad(outdir / "GSE224273_merged_rna_raw.h5ad").copy()
sc.pp.normalize_total(plaque, target_sum=1e4)
sc.pp.log1p(plaque)

nk_seed = [g for g in ["NKG7","GNLY","KLRD1","FCGR3A","TYROBP"] if g in plaque.var_names]
sc.tl.score_genes(plaque, nk_seed, score_name="nk_seed_score", use_raw=False)
thresh = plaque.obs["nk_seed_score"].quantile(0.90)
plaque.obs["is_nk_high"] = plaque.obs["nk_seed_score"] >= thresh
plaque = plaque[plaque.obs["is_nk_high"]].copy()

pathways = {
    "Cytotoxic / NK effector": ["NKG7","GNLY","PRF1","GZMB","CTSW","KLRD1","FCGR3A","TYROBP","XCL1","XCL2"],
    "Inflammatory / cytokine": ["CCL3","CCL4","CCL5","IL32","TNF","NFKBIA","JUN","FOS","DUSP1","LTB","IFNG"],
    "Interferon response": ["IFIT1","IFIT2","IFIT3","ISG15","MX1","OAS1","IFI6","IFITM1","IFITM2","IFITM3","STAT1"],
    "Migration / adhesion": ["CX3CR1","ITGAL","ITGB2","SELL","CCR7","CXCR3","CCL5","ICAM3","RAC2"],
    "Antigen presentation": ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAPBP","CD74","HLA-DRA","HLA-DRB1"],
    "Lipid / vascular relevance": ["FCGR3A","TYROBP","CX3CR1","NFKBIA","CCL5","LTB","IFITM3","JUN","FOS","NR4A1","ITGB2"]
}

# PBMC signature effects
pbmc_rows = []
for name, genes in pathways.items():
    genes_use = [g for g in genes if g in pbmc.var_names]
    if len(genes_use) < 2:
        pbmc_rows.append({"pathway": name, "dataset": "PBMC", "effect": np.nan, "n_genes": len(genes_use)})
        continue
    sc.tl.score_genes(pbmc, genes_use, score_name="_tmp_score", use_raw=False)
    mean1 = pbmc.obs.loc[pbmc.obs["leiden_r06"]=="1", "_tmp_score"].mean()
    mean4 = pbmc.obs.loc[pbmc.obs["leiden_r06"]=="4", "_tmp_score"].mean()
    pbmc_rows.append({"pathway": name, "dataset": "PBMC", "effect": mean1 - mean4, "n_genes": len(genes_use)})
    del pbmc.obs["_tmp_score"]

# Plaque signature effects (sample-level asym - sym)
plaque_rows = []
for name, genes in pathways.items():
    genes_use = [g for g in genes if g in plaque.var_names]
    if len(genes_use) < 2:
        plaque_rows.append({"pathway": name, "dataset": "Plaque", "effect": np.nan, "n_genes": len(genes_use)})
        continue
    sc.tl.score_genes(plaque, genes_use, score_name="_tmp_score", use_raw=False)
    obs = plaque.obs.copy()
    obs["_tmp_score"] = plaque.obs["_tmp_score"].values
    if "status" not in obs.columns:
        plaque_rows.append({"pathway": name, "dataset": "Plaque", "effect": np.nan, "n_genes": len(genes_use)})
        del plaque.obs["_tmp_score"]
        continue
    sample_cols = [c for c in ["sample", "gsm", "status"] if c in obs.columns]
    samp = obs.groupby(sample_cols, observed=True)["_tmp_score"].mean().reset_index()
    a = samp.loc[samp["status"]=="Asymptomatic","_tmp_score"].astype(float).values
    s = samp.loc[samp["status"]=="Symptomatic","_tmp_score"].astype(float).values
    effect = np.nan if len(a)==0 or len(s)==0 else a.mean() - s.mean()
    plaque_rows.append({"pathway": name, "dataset": "Plaque", "effect": effect, "n_genes": len(genes_use)})
    del plaque.obs["_tmp_score"]

df = pd.DataFrame(pbmc_rows + plaque_rows)
df.to_csv(outdir / "Figure3_cross_dataset_pathway_concordance.csv", index=False)

mat = df.pivot(index="pathway", columns="dataset", values="effect")
mat = mat.reindex(list(pathways.keys()))

mat_z = mat.copy()
for c in mat_z.columns:
    vals = mat_z[c].values.astype(float)
    if np.nanstd(vals) == 0 or np.isnan(np.nanstd(vals)):
        mat_z[c] = 0
    else:
        mat_z[c] = (vals - np.nanmean(vals)) / np.nanstd(vals)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(5.4, 6.0))
im = ax.imshow(mat_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)

ax.set_title("M", loc="left", fontweight="bold", fontsize=18)
ax.set_xticks(np.arange(len(mat_z.columns)))
ax.set_xticklabels(mat_z.columns.tolist())
ax.set_yticks(np.arange(len(mat_z.index)))
ax.set_yticklabels(mat_z.index.tolist())

for i, pth in enumerate(mat.index):
    for j, ds in enumerate(mat.columns):
        val = mat.loc[pth, ds]
        txt = "NA" if pd.isna(val) else f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.2)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("Relative pathway effect")

png = figdir / "PanelM_Figure3_cross_dataset_pathway_concordance.png"
pdf = figdir / "PanelM_Figure3_cross_dataset_pathway_concordance.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(mat)
