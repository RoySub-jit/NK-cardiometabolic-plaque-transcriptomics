from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

genes_core = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1"] if g in nk.var_names]

expr = nk[:, genes_core].to_df()
obs = nk.obs.copy()

needed_obs = []
for c in ["sample", "gsm", "status"]:
    if c in obs.columns:
        needed_obs.append(c)

expr = pd.concat([obs[needed_obs].reset_index(drop=True), expr.reset_index(drop=True)], axis=1)

group_cols = [c for c in ["sample", "gsm", "status"] if c in expr.columns]
sample_summary = (
    expr.groupby(group_cols, observed=True)[genes_core]
    .mean()
    .reset_index()
)

sample_summary.to_csv(outdir / "GSE224273_nk_high_core_gene_sample_summary.csv", index=False)

status_order = ["Asymptomatic", "Symptomatic"]
fills = {"Asymptomatic": "#4C78A8", "Symptomatic": "#E45756"}

def exact_permutation_test(x, y, n_perm=10000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    nx = len(x)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:nx].mean() - perm[nx:].mean()
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p

def bootstrap_ci(x, y, n_boot=5000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xb.mean() - yb.mean()
    return np.quantile(diffs, [0.025, 0.975])

# stats table
stat_rows = []
for gene in genes_core:
    a = pd.to_numeric(sample_summary.loc[sample_summary["status"] == "Asymptomatic", gene], errors="coerce").dropna().values
    s = pd.to_numeric(sample_summary.loc[sample_summary["status"] == "Symptomatic", gene], errors="coerce").dropna().values
    diff, p = exact_permutation_test(a, s, n_perm=10000, seed=42)
    ci_low, ci_high = bootstrap_ci(a, s, n_boot=5000, seed=42)
    stat_rows.append({
        "gene": gene,
        "asym_mean": a.mean(),
        "sym_mean": s.mean(),
        "asym_minus_sym_mean": diff,
        "exact_permutation_p": p,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "n_asym": len(a),
        "n_sym": len(s),
    })

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(outdir / "GSE224273_nk_high_core_gene_stats.csv", index=False)

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

fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6))
axes = axes.flatten()
rng = np.random.default_rng(42)

letters = ["G", "H", "I", "J"]

for ax, gene, letter in zip(axes, genes_core, letters):
    a = pd.to_numeric(sample_summary.loc[sample_summary["status"] == "Asymptomatic", gene], errors="coerce").dropna().values
    s = pd.to_numeric(sample_summary.loc[sample_summary["status"] == "Symptomatic", gene], errors="coerce").dropna().values

    parts = ax.violinplot(
        [a, s],
        positions=[1, 2],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for i, body in enumerate(parts["bodies"]):
        status = status_order[i]
        body.set_facecolor(fills[status])
        body.set_edgecolor("black")
        body.set_alpha(0.25)
        body.set_linewidth(0.8)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    for i, (vals, status) in enumerate(zip([a, s], status_order), start=1):
        x = i + rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(x, vals, s=55, color=fills[status], edgecolors="black", linewidths=0.55, zorder=3)

    st = stats_df.loc[stats_df["gene"] == gene].iloc[0]
    ax.text(
        0.03, 0.97,
        f"P = {st['exact_permutation_p']:.4f}\n"
        f"Δ = {st['asym_minus_sym_mean']:.3f}\n"
        f"95% CI [{st['bootstrap_ci_low']:.3f}, {st['bootstrap_ci_high']:.3f}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(status_order)
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("NK-high plaque cells: core GZMK-like genes by plaque group", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(figdir / "PanelGJ_GSE224273_NKhigh_core_genes_violin.png", dpi=400, bbox_inches="tight")
fig.savefig(figdir / "PanelGJ_GSE224273_NKhigh_core_genes_violin.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:", figdir / "PanelGJ_GSE224273_NKhigh_core_genes_violin.png")
print("Saved:", figdir / "PanelGJ_GSE224273_NKhigh_core_genes_violin.pdf")
print("Saved stats:", outdir / "GSE224273_nk_high_core_gene_stats.csv")
print(stats_df.to_string(index=False))
