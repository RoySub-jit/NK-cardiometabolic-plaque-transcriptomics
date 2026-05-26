from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE224273_merged_rna_raw.h5ad").copy()

# normalize/log
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# define NK-high cells
nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]
nk_seed_genes = [g for g in nk_seed_genes if g in adata.var_names]
sc.tl.score_genes(adata, nk_seed_genes, score_name="nk_seed_score", use_raw=False)
thresh = adata.obs["nk_seed_score"].quantile(0.90)
adata.obs["is_nk_high"] = adata.obs["nk_seed_score"] >= thresh

nk = adata[adata.obs["is_nk_high"]].copy()

genes_core = ["GZMK", "NKG7", "GNLY", "KLRD1"]
genes_ext = ["PRF1", "GZMB"]
genes_all = [g for g in genes_core + genes_ext if g in nk.var_names]

expr = nk[:, genes_all].to_df()
obs = nk.obs.copy()

needed_obs = []
for c in ["sample", "gsm", "status"]:
    if c in obs.columns:
        needed_obs.append(c)

expr = pd.concat([obs[needed_obs].reset_index(drop=True), expr.reset_index(drop=True)], axis=1)

# ---------- save cell-level table ----------
expr.to_csv(outdir / "GSE224273_nk_high_gene_cell_level.csv", index=False)

# ---------- sample-level summaries ----------
group_cols = [c for c in ["sample", "gsm", "status"] if c in expr.columns]
sample_summary = (
    expr.groupby(group_cols, observed=True)[genes_all]
    .mean()
    .reset_index()
)
sample_summary.to_csv(outdir / "GSE224273_nk_high_gene_sample_summary.csv", index=False)

# ---------- correlations inside NK-high cells ----------
corr_pairs = [("GZMK", "NKG7"), ("GZMK", "GNLY"), ("GZMK", "KLRD1")]
corr_rows = []
for g1, g2 in corr_pairs:
    if g1 in expr.columns and g2 in expr.columns:
        rho, p = spearmanr(expr[g1], expr[g2], nan_policy="omit")
        corr_rows.append({"gene_x": g1, "gene_y": g2, "rho": rho, "p_value": p})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(outdir / "GSE224273_nk_high_gene_correlations.csv", index=False)

print("Saved sample summary:", outdir / "GSE224273_nk_high_gene_sample_summary.csv")
print("Saved correlations:", outdir / "GSE224273_nk_high_gene_correlations.csv")

# ---------- Panel G: box/jitter per sample ----------
status_order = ["Asymptomatic", "Symptomatic"]
fills = {"Asymptomatic": "#4C78A8", "Symptomatic": "#E45756"}

plot_genes = [g for g in genes_core if g in sample_summary.columns]

fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
axes = axes.flatten()
rng = np.random.default_rng(42)

for ax, gene, letter in zip(axes, plot_genes, ["G", "H", "I", "J"]):
    groups = []
    for s in status_order:
        vals = pd.to_numeric(sample_summary.loc[sample_summary["status"] == s, gene], errors="coerce").dropna().values
        groups.append(vals)

    bp = ax.boxplot(
        groups,
        positions=[1, 2],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, s in zip(bp["boxes"], status_order):
        patch.set_facecolor(fills[s])
        patch.set_alpha(0.28)
        patch.set_edgecolor("black")

    for i, s in enumerate(status_order, start=1):
        vals = pd.to_numeric(sample_summary.loc[sample_summary["status"] == s, gene], errors="coerce").dropna().values
        x = i + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x, vals, s=50, color=fills[s], edgecolors="black", linewidths=0.55, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(status_order)
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("NK-high plaque cells: core GZMK-like genes by plaque group", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(figdir / "PanelGJ_GSE224273_NKhigh_core_genes_boxjitter.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "PanelGJ_GSE224273_NKhigh_core_genes_boxjitter.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Panel K: gene-gene correlations ----------
pairs_to_plot = [p for p in corr_pairs if p[0] in expr.columns and p[1] in expr.columns]
fig, axes = plt.subplots(1, len(pairs_to_plot), figsize=(5.2 * len(pairs_to_plot), 4.8))
if len(pairs_to_plot) == 1:
    axes = [axes]

for ax, (g1, g2), letter in zip(axes, pairs_to_plot, ["K", "L", "M"]):
    x = pd.to_numeric(expr[g1], errors="coerce")
    y = pd.to_numeric(expr[g2], errors="coerce")
    rho, p = spearmanr(x, y, nan_policy="omit")

    ax.scatter(x, y, s=10, color="#2f78b7", alpha=0.35, edgecolors="none")
    ax.set_xlabel(g1)
    ax.set_ylabel(g2)
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.03, 0.97,
        f"ρ = {rho:.3f}\nP = {p:.4g}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
    )

fig.suptitle("NK-high plaque cells: gene-gene coherence", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(figdir / "PanelK_M_GSE224273_NKhigh_gene_correlations.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "PanelK_M_GSE224273_NKhigh_gene_correlations.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Panel N: sample heatmap ----------
heat_genes = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"] if g in sample_summary.columns]
heat = sample_summary.copy()

# make readable row labels
if "sample" in heat.columns and "status" in heat.columns:
    heat["row_label"] = heat["sample"].astype(str) + " | " + heat["status"].astype(str)
elif "gsm" in heat.columns and "status" in heat.columns:
    heat["row_label"] = heat["gsm"].astype(str) + " | " + heat["status"].astype(str)
else:
    heat["row_label"] = np.arange(len(heat)).astype(str)

heat = heat.sort_values(["status", "row_label"]).reset_index(drop=True)
mat = heat[heat_genes].copy()
mat = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0)

fig, ax = plt.subplots(figsize=(7.6, max(4.2, 0.42 * len(heat))))
im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=-1.5, vmax=1.5)

ax.set_xticks(np.arange(len(heat_genes)))
ax.set_xticklabels(heat_genes, rotation=35, ha="right")
ax.set_yticks(np.arange(len(heat)))
ax.set_yticklabels(heat["row_label"].tolist())
ax.set_title("N", loc="left", fontweight="bold", fontsize=16)
ax.set_xlabel("Genes")
ax.set_ylabel("NK-high plaque samples")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Relative mean expression")

fig.suptitle("NK-high plaque gene decomposition across samples", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(figdir / "PanelN_GSE224273_NKhigh_sample_heatmap.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "PanelN_GSE224273_NKhigh_sample_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved figure panels to:", figdir)
