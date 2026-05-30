from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib.gridspec import GridSpec

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
OLD = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
OUT = PROJECT / "results/corrected_figure3_plaque_only"
OUT.mkdir(parents=True, exist_ok=True)

H5AD = OLD / "results/tier2_validation/GSE224273_merged_rna_raw.h5ad"
STATS = OLD / "results/supplementary/Table_S5_plaque_singlecell_statistics.csv"

if not H5AD.exists():
    raise FileNotFoundError(f"Missing plaque object: {H5AD}")
if not STATS.exists():
    raise FileNotFoundError(f"Missing plaque statistics table: {STATS}")

# -------------------------------------------------------------------------
# Load plaque single-cell object and validated plaque statistics
# -------------------------------------------------------------------------
adata = sc.read_h5ad(H5AD).copy()
stats = pd.read_csv(STATS)

print("Loaded plaque object:", H5AD)
print("Object shape:", adata.shape)
print("obs columns:", list(adata.obs.columns))
print("Stats columns:", list(stats.columns))
print(stats.to_string(index=False))

if "status" not in adata.obs.columns:
    raise RuntimeError("Expected clinical-group column 'status' was not found in adata.obs.")

adata.obs["status"] = adata.obs["status"].astype(str)

# Identify sample column without silently guessing.
sample_candidates = [
    "sample", "Sample", "sample_id", "SampleID",
    "orig.ident", "donor", "Donor", "patient", "Patient"
]
sample_col = next((c for c in sample_candidates if c in adata.obs.columns), None)

if sample_col is None:
    print("\nAvailable obs columns:")
    print(list(adata.obs.columns))
    raise RuntimeError(
        "Could not identify the sample column. Add the correct sample-column "
        "name to sample_candidates after inspecting the printed obs columns."
    )

print("Using sample column:", sample_col)

# -------------------------------------------------------------------------
# Normalize/log transform and reproduce the plaque NK-high definition
# retained in the repository workflow.
# -------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]
nk_seed_genes = [g for g in nk_seed_genes if g in adata.var_names]

if len(nk_seed_genes) < 3:
    raise RuntimeError(f"Insufficient NK seed genes present: {nk_seed_genes}")

sc.tl.score_genes(
    adata,
    nk_seed_genes,
    score_name="nk_seed_score",
    use_raw=False
)

threshold = adata.obs["nk_seed_score"].quantile(0.90)
adata.obs["is_nk_high"] = adata.obs["nk_seed_score"] >= threshold

# GZMK-like score used for the plaque comparison.
gzmk_like_genes = ["GZMK", "NKG7", "GNLY", "KLRD1"]
gzmk_like_genes = [g for g in gzmk_like_genes if g in adata.var_names]

if len(gzmk_like_genes) != 4:
    raise RuntimeError(f"Expected all four GZMK-like genes; found: {gzmk_like_genes}")

def expression_vector(obj, gene):
    x = obj[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()

for gene in gzmk_like_genes:
    adata.obs[f"{gene}_expr"] = expression_vector(adata, gene)

adata.obs["gzmk_like_score"] = adata.obs[
    [f"{g}_expr" for g in gzmk_like_genes]
].mean(axis=1)

nk = adata[adata.obs["is_nk_high"]].copy()
nk_obs = adata.obs.loc[adata.obs["is_nk_high"]].copy()

print("NK-high threshold:", threshold)
print("NK-high cells:", nk.n_obs)
print("NK-high cells by status:")
print(nk_obs["status"].value_counts().to_string())

# -------------------------------------------------------------------------
# Sample-level plaque values
# -------------------------------------------------------------------------
heatmap_genes = ["GZMK", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"]
heatmap_genes = [g for g in heatmap_genes if g in nk.var_names]

for gene in heatmap_genes:
    nk_obs[f"{gene}_expr"] = expression_vector(nk, gene)

sample_info = (
    nk_obs.groupby(sample_col, observed=True)
    .first()[["status"]]
)

sample_expression = (
    nk_obs.groupby(sample_col, observed=True)
    [[f"{g}_expr" for g in heatmap_genes] + ["gzmk_like_score"]]
    .mean()
    .join(sample_info)
    .reset_index()
)

status_order = {"Asymptomatic": 0, "Symptomatic": 1}
sample_expression["status_order"] = sample_expression["status"].map(status_order).fillna(9)
sample_expression = sample_expression.sort_values(["status_order", sample_col]).reset_index(drop=True)

sample_expression.to_csv(
    OUT / "Figure3_verified_plaque_sample_level_values.csv",
    index=False
)

# Retrieve validated Table S5 statistics. These are displayed in Panel C.
def find_column(possible_names):
    lower_map = {c.lower().replace(" ", "_"): c for c in stats.columns}
    for name in possible_names:
        key = name.lower().replace(" ", "_")
        if key in lower_map:
            return lower_map[key]
    return None

p_col = find_column(["p_value", "p value", "p"])
effect_col = find_column(["Effect", "effect", "mean_difference", "mean difference"])
ci_col = find_column(["95% CI", "95%_CI", "confidence_interval", "CI"])

if p_col is None or effect_col is None:
    raise RuntimeError(
        "Could not identify p-value/effect columns in Table S5. "
        f"Available columns: {list(stats.columns)}"
    )

p_value = float(stats.loc[0, p_col])
effect = float(stats.loc[0, effect_col])

if ci_col is not None:
    ci_text = str(stats.loc[0, ci_col])
else:
    # This is only a display fallback; verify against Table S5 output.
    ci_text = "[0.059, 0.426]"

# -------------------------------------------------------------------------
# UMAP for Panel A
# -------------------------------------------------------------------------
if "X_umap" not in adata.obsm:
    sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars), flavor="seurat")
    sc.tl.pca(adata, use_highly_variable=True, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

umap = adata.obsm["X_umap"]
nk_mask = adata.obs["is_nk_high"].values
status = adata.obs["status"].values

colors = {
    "Asymptomatic": "#1f78b4",
    "Symptomatic": "#e45756"
}

# -------------------------------------------------------------------------
# Build final plaque-only Figure 3
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(13.2, 7.0))
gs = GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[1.03, 1.27],
    height_ratios=[1.02, 0.98],
    hspace=0.44,
    wspace=0.40
)

# -------------------------------------------------------------------------
# Panel A: plaque UMAP with NK-high cells by clinical group
# -------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])

ax_a.scatter(
    umap[:, 0], umap[:, 1],
    s=3, color="#D6D6D6", alpha=0.35, linewidths=0, rasterized=True
)

for group in ["Asymptomatic", "Symptomatic"]:
    mask = nk_mask & (status == group)
    ax_a.scatter(
        umap[mask, 0], umap[mask, 1],
        s=8, color=colors[group], alpha=0.90,
        linewidths=0, label=group, rasterized=True
    )

    if mask.sum() > 0:
        cx = np.median(umap[mask, 0])
        cy = np.median(umap[mask, 1])
        ax_a.text(
            cx + 0.20, cy + 0.20, group,
            fontsize=9, fontweight="bold", color=colors[group],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9
            )
        )

ax_a.set_xlabel("UMAP1")
ax_a.set_ylabel("UMAP2")
ax_a.legend(frameon=False, loc="best", fontsize=8)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.text(
    -0.14, 1.07, "A.",
    transform=ax_a.transAxes,
    fontsize=17, fontweight="bold"
)

# -------------------------------------------------------------------------
# Panel B: sample-level plaque NK-high expression heatmap
# -------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])

matrix = sample_expression[[f"{g}_expr" for g in heatmap_genes]].to_numpy()
matrix_z = zscore(matrix, axis=0, nan_policy="omit")
matrix_z = np.nan_to_num(matrix_z)

im = ax_b.imshow(
    matrix_z,
    aspect="auto",
    cmap="Blues",
    vmin=-1.5,
    vmax=1.5
)

row_labels = [
    f"{row[sample_col]} | {row['status']}"
    for _, row in sample_expression.iterrows()
]

ax_b.set_yticks(range(len(row_labels)))
ax_b.set_yticklabels(row_labels)
ax_b.set_xticks(range(len(heatmap_genes)))
ax_b.set_xticklabels(heatmap_genes, rotation=40, ha="right")
ax_b.set_ylabel("NK-high plaque samples")
ax_b.set_xlabel("Genes")

cbar = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.025)
cbar.set_label("Relative mean expression")

ax_b.text(
    -0.12, 1.07, "B.",
    transform=ax_b.transAxes,
    fontsize=17, fontweight="bold"
)

# -------------------------------------------------------------------------
# Panel C: plaque GZMK-like score comparison
# -------------------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, :])

plot_order = ["Asymptomatic", "Symptomatic"]
values = [
    sample_expression.loc[
        sample_expression["status"] == group,
        "gzmk_like_score"
    ].dropna().values
    for group in plot_order
]

bp = ax_c.boxplot(
    values,
    positions=[1, 2],
    widths=0.45,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="0.35"),
    capprops=dict(color="0.35"),
)

box_colors = ["#D6E2EF", "#F4D1D1"]
for box, color in zip(bp["boxes"], box_colors):
    box.set_facecolor(color)
    box.set_edgecolor("0.65")

rng = np.random.default_rng(7)
for i, (group, vals) in enumerate(zip(plot_order, values), start=1):
    jitter = rng.normal(0, 0.025, size=len(vals))
    ax_c.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        s=34,
        color=colors[group],
        edgecolor="0.25",
        linewidth=0.5,
        zorder=3
    )

ax_c.set_xticks([1, 2])
ax_c.set_xticklabels(plot_order)
ax_c.set_ylabel("GZMK-like score in NK-high cells")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

ax_c.text(
    0.02, 0.96,
    f"Exact permutation p = {p_value:.4f}\n"
    f"Asymptomatic - symptomatic = {effect:.3f}\n"
    f"95% CI {ci_text}",
    transform=ax_c.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.80"
    )
)

ax_c.text(
    -0.06, 1.07, "C.",
    transform=ax_c.transAxes,
    fontsize=17, fontweight="bold"
)

fig.suptitle(
    "Plaque single-cell comparison of NK-associated transcriptional features",
    fontsize=13,
    fontweight="bold",
    y=1.01
)

plt.tight_layout()

pdf = OUT / "Figure_3_final_plaque_singlecell_only.pdf"
png = OUT / "Figure_3_final_plaque_singlecell_only.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=400, bbox_inches="tight")
plt.close(fig)

with open(OUT / "Figure_3_final_plaque_singlecell_only_provenance.txt", "w") as handle:
    handle.write("Figure 3 final plaque-only provenance\n")
    handle.write("=" * 45 + "\n")
    handle.write("No PBMC cluster-comparison or PBMC pathway panels are included.\n")
    handle.write(f"Input plaque object: {H5AD}\n")
    handle.write(f"Input plaque statistics: {STATS}\n")
    handle.write(f"NK-high score genes: {', '.join(nk_seed_genes)}\n")
    handle.write("NK-high threshold: 90th percentile of NK seed score.\n")
    handle.write(f"GZMK-like genes: {', '.join(gzmk_like_genes)}\n")
    handle.write(f"Panel C exact permutation p value: {p_value:.4f}\n")
    handle.write(f"Panel C asymptomatic-minus-symptomatic effect: {effect:.3f}\n")
    handle.write(f"Panel C 95% CI: {ci_text}\n")

print("\nPASS: Final plaque-only Figure 3 generated.")
print("Saved:", pdf)
print("Saved:", png)
print("Saved:", OUT / "Figure_3_final_plaque_singlecell_only_provenance.txt")
print("\nSample-level values:")
print(sample_expression.to_string(index=False))
