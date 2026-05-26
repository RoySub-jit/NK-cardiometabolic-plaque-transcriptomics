
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
outdir = base / "results" / "tier2_validation"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE198339_merged_singlecell_umap.h5ad")
meta = pd.read_csv(base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_validation_participant_level.csv")

clusters_keep = ["1", "4", "10"]
obs = adata.obs.copy()
obs["leiden_r06"] = obs["leiden_r06"].astype(str)

# participant-level cluster abundance
tab = pd.crosstab(obs["participant_id"], obs["leiden_r06"], normalize="index")
for c in clusters_keep:
    if c not in tab.columns:
        tab[c] = 0.0

tab["nk_enriched_combined"] = tab[clusters_keep].sum(axis=1)
tab = tab.reset_index()

# merge with metadata
m = meta[["ParticipantID", "non_HDL"]].copy().drop_duplicates()
m = m.rename(columns={"ParticipantID": "participant_id"})
m["non_HDL"] = pd.to_numeric(m["non_HDL"], errors="coerce")

df = tab.merge(m, on="participant_id", how="inner").dropna(subset=["non_HDL"]).copy()
for c in ["1", "4", "10", "nk_enriched_combined"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.to_csv(outdir / "GSE198339_nk_cluster_abundance_by_participant.csv", index=False)

x = df["non_HDL"].values
y = df["nk_enriched_combined"].values

rho, p = spearmanr(x, y)
lr = linregress(x, y)
xline = np.linspace(x.min(), x.max(), 200)
yline = lr.intercept + lr.slope * xline

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

fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(x, y, s=42, color="#2f78b7", edgecolors="black", linewidths=0.5, alpha=0.9)
ax.plot(xline, yline, color="#1f4e79", linewidth=1.5)

ax.set_title("H", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Non-HDL cholesterol")
ax.set_ylabel("Combined NK-enriched cluster proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(
    0.03, 0.97,
    f"Spearman ρ = {rho:.3f}\nP = {p:.4f}\nn = {len(df)}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelH_GSE198339_NKcluster_abundance_scatter.png"
pdf = figdir / "PanelH_GSE198339_NKcluster_abundance_scatter.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(df[["participant_id", "non_HDL", "1", "4", "10", "nk_enriched_combined"]].to_string(index=False))
print("Spearman rho:", rho)
print("P value:", p)
