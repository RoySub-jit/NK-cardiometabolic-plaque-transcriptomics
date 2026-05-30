from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import zscore, linregress

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
OBJ = PROJECT / "results/corrected_object/GSE198339_official_processed_9368cells_normalized_umap.h5ad"
STATS = PROJECT / "results/corrected_pbmc_statistics/GSE198339_verified_nonHDL_NK_statistics.csv"
METRICS = PROJECT / "results/corrected_pbmc_statistics/GSE198339_verified_participant_level_NK_metrics.csv"
OUT = PROJECT / "results/corrected_figure_panels"
OUT.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(OBJ)
stats = pd.read_csv(STATS)
metrics = pd.read_csv(METRICS)

nk_labels = ["NK cells resting", "NK cells activated"]
nk = adata[adata.obs["ClusterName"].isin(nk_labels)].copy()
genes = ["NKG7", "GZMK", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRD1", "TYROBP"]

def get_expr(a, gene):
    x = a[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Panel B: verified UMAP with official NK annotations highlighted
# ------------------------------------------------------------
umap = adata.obsm["X_umap"]
obs = adata.obs.copy()

fig, ax = plt.subplots(figsize=(5.5, 4.8))
ax.scatter(umap[:, 0], umap[:, 1], s=1.5, alpha=0.25, rasterized=True)

for label, marker in zip(nk_labels, ["o", "^"]):
    mask = obs["ClusterName"].astype(str) == label
    ax.scatter(
        umap[mask, 0], umap[mask, 1],
        s=5, marker=marker, label=label, rasterized=True
    )

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(frameon=False, fontsize=8, markerscale=2)
ax.set_title("Official GSE198339 annotations")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "Fig2B_verified_UMAP")

# ------------------------------------------------------------
# Panel C: participant NK composition ordered by non-HDL
# ------------------------------------------------------------
obs["ParticipantID"] = obs["ParticipantID"].astype(str)
obs["ClusterName"] = obs["ClusterName"].astype(str)

totals = obs.groupby("ParticipantID", observed=True).size()
nk_comp = (
    obs[obs["ClusterName"].isin(nk_labels)]
    .groupby(["ParticipantID", "ClusterName"], observed=True)
    .size()
    .unstack(fill_value=0)
)
for label in nk_labels:
    if label not in nk_comp.columns:
        nk_comp[label] = 0

nk_prop = nk_comp[nk_labels].div(totals, axis=0)
clinical = metrics.set_index("ParticipantID")[["non_HDL"]]
plot_df = nk_prop.join(clinical).sort_values("non_HDL")

fig, ax = plt.subplots(figsize=(6.0, 4.5))
bottom = np.zeros(len(plot_df))
for label in nk_labels:
    vals = plot_df[label].values
    ax.bar(plot_df.index, vals, bottom=bottom, label=label)
    bottom += vals

ax.set_ylabel("Proportion of total PBMCs")
ax.set_xlabel("Participants ordered by non-HDL cholesterol")
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df.index.str.replace("Participant_", "P"), rotation=45, ha="right")
for i, val in enumerate(plot_df["non_HDL"]):
    ax.text(i, bottom[i] + 0.003, str(int(val)), ha="center", va="bottom", fontsize=7)
ax.text(0.02, 0.96, "Values above bars: non-HDL", transform=ax.transAxes, fontsize=8, va="top")
ax.legend(frameon=False, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "Fig2C_verified_NK_composition")

# ------------------------------------------------------------
# Panel D: marker heatmap across official NK annotations
# ------------------------------------------------------------
mean_rows = []
for label in nk_labels:
    subset = adata[adata.obs["ClusterName"].astype(str) == label]
    row = [get_expr(subset, gene).mean() for gene in genes]
    mean_rows.append(row)

heat = np.asarray(mean_rows)
heat_scaled = zscore(heat, axis=0, nan_policy="omit")
heat_scaled = np.nan_to_num(heat_scaled)

fig, ax = plt.subplots(figsize=(6.0, 2.7))
im = ax.imshow(heat_scaled, aspect="auto")
ax.set_xticks(range(len(genes)))
ax.set_xticklabels(genes, rotation=45, ha="right")
ax.set_yticks(range(len(nk_labels)))
ax.set_yticklabels(nk_labels)
ax.set_title("Relative mean expression in annotated NK populations")
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Scaled mean expression")
save(fig, "Fig2D_verified_NK_marker_heatmap")

# ------------------------------------------------------------
# Scatter plot helper: Panels E-G
# ------------------------------------------------------------
def scatter_panel(feature_column, ylabel, rho, p_value, stem):
    df = metrics[["non_HDL", feature_column]].dropna().sort_values("non_HDL")
    slope, intercept, _, _, _ = linregress(df["non_HDL"], df[feature_column])
    xline = np.linspace(df["non_HDL"].min(), df["non_HDL"].max(), 100)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(df["non_HDL"], df[feature_column], s=34)
    ax.plot(xline, intercept + slope * xline, linewidth=1.2)
    ax.set_xlabel("Non-HDL cholesterol")
    ax.set_ylabel(ylabel)
    ax.text(
        0.04, 0.95,
        f"Spearman \u03c1 = {rho:.3f}\np = {p_value:.5f}\nn = 8",
        transform=ax.transAxes, ha="left", va="top", fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, stem)

scatter_panel(
    "Cytotoxic_core_score",
    "Cytotoxic core score",
    0.874267, 0.004512,
    "Fig2E_verified_cytotoxic_core_nonHDL"
)

scatter_panel(
    "NKG7_expr",
    "NKG7 expression in NK cells",
    0.838338, 0.009323,
    "Fig2F_verified_NKG7_nonHDL"
)

scatter_panel(
    "NK_resting_proportion",
    "NK resting-cell proportion",
    0.826362, 0.011443,
    "Fig2G_verified_NK_resting_nonHDL"
)

# ------------------------------------------------------------
# Panel H: participant-level NK expression heatmap
# ------------------------------------------------------------
display_cols = [f"{g}_expr" for g in genes]
display_names = genes

ordered = metrics.sort_values("non_HDL").copy()
matrix = ordered[display_cols].to_numpy()
matrix_scaled = zscore(matrix, axis=0, nan_policy="omit")
matrix_scaled = np.nan_to_num(matrix_scaled)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(matrix_scaled.T, aspect="auto")
ax.set_xticks(range(len(ordered)))
ax.set_xticklabels(
    [f"P{x.split('_')[-1]}\n({int(v)})" for x, v in zip(ordered["ParticipantID"], ordered["non_HDL"])],
    rotation=45, ha="right"
)
ax.set_yticks(range(len(display_names)))
ax.set_yticklabels(display_names)
ax.set_xlabel("Participants ordered by non-HDL (value in parentheses)")
ax.set_title("NK-cell expression features")
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Relative expression")
save(fig, "Fig2H_verified_participant_heatmap")

# ------------------------------------------------------------
# Panel I: verified correlation summary
# ------------------------------------------------------------
summary_features = [
    ("Cytotoxic core score", "Cytotoxic core score"),
    ("NKG7 expression in NK cells", "NKG7 expression"),
    ("NK resting-cell proportion", "NK resting-cell proportion"),
    ("GZMK-like composite score", "GZMK-like score"),
]

values = []
labels = []
annotations = []
for raw_label, display_label in summary_features:
    row = stats.loc[stats["Feature"] == raw_label].iloc[0]
    values.append(row["Spearman_rho"])
    labels.append(display_label)
    annotations.append(f"\u03c1={row['Spearman_rho']:.3f}\np={row['p_value']:.4f}")

arr = np.asarray(values).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(4.6, 4.1))
im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xticks([0])
ax.set_xticklabels(["Spearman \u03c1"])
for i, txt in enumerate(annotations):
    ax.text(0, i, txt, ha="center", va="center", fontsize=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.05)
cbar.set_label("Correlation strength")
ax.set_title("Verified PBMC disease-context associations")
save(fig, "Fig2I_verified_summary_heatmap")

# ------------------------------------------------------------
# Supplementary Fig. S1: corrected summary plot
# ------------------------------------------------------------
s1_labels = [
    "Cytotoxic core score",
    "NKG7 expression in NK cells",
    "NK resting-cell proportion",
    "GZMK-like composite score",
]
s1_rho = [0.874267, 0.838338, 0.826362, 0.658694]
s1_p = [0.004512, 0.009323, 0.011443, 0.075690]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
y = np.arange(len(s1_labels))[::-1]
bars = ax.barh(y, s1_rho)
ax.set_yticks(y)
ax.set_yticklabels(s1_labels)
ax.set_xlabel("Spearman \u03c1")
ax.set_xlim(0, 1.05)
for yi, rho, p in zip(y, s1_rho, s1_p):
    ax.text(rho + 0.02, yi, f"\u03c1={rho:.3f}, p={p:.4f}", va="center", fontsize=9)
ax.set_title("Fig. S1. Verified PBMC disease-context association summary")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "Supplementary_Fig_S1_verified_PBMC_summary")

# Export compact data audit for figure replacement.
figure_stats = pd.DataFrame({
    "Figure_feature": s1_labels,
    "Spearman_rho": s1_rho,
    "p_value": s1_p,
    "Status": [
        "Significant positive association",
        "Significant positive association",
        "Significant positive association",
        "Positive non-significant trend",
    ]
})
figure_stats.to_csv(OUT / "verified_PBMC_figure_statistics.csv", index=False)

print("PASS: Corrected PBMC figure panels generated.")
for f in sorted(OUT.glob("*")):
    print(f.name)
