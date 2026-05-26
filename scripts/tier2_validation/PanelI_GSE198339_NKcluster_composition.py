
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
outdir = base / "results" / "tier2_validation"
figdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(outdir / "GSE198339_merged_singlecell_umap.h5ad")
meta = pd.read_csv(base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_validation_participant_level.csv")

clusters_keep = ["1", "4", "10"]
obs = adata.obs.copy()
obs["leiden_r06"] = obs["leiden_r06"].astype(str)

tab = pd.crosstab(obs["participant_id"], obs["leiden_r06"], normalize="index")
for c in clusters_keep:
    if c not in tab.columns:
        tab[c] = 0.0
tab = tab[clusters_keep].reset_index()

m = meta[["ParticipantID", "non_HDL"]].copy().drop_duplicates()
m = m.rename(columns={"ParticipantID": "participant_id"})
m["non_HDL"] = pd.to_numeric(m["non_HDL"], errors="coerce")

df = tab.merge(m, on="participant_id", how="inner").dropna(subset=["non_HDL"]).copy()
df = df.sort_values("non_HDL").reset_index(drop=True)

colors = {"1": "#4daf4a", "4": "#1f78b4", "10": "#984ea3"}

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8.0, 5.4))

bottom = np.zeros(len(df))
for c in clusters_keep:
    vals = df[c].values
    ax.bar(np.arange(len(df)), vals, bottom=bottom, color=colors[c], edgecolor="white", linewidth=0.4, label=f"Cluster {c}")
    bottom += vals

ax.set_title("I", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Participants ordered by non-HDL")
ax.set_ylabel("Proportion of cells")
ax.set_xticks(np.arange(len(df)))
ax.set_xticklabels(df["participant_id"].tolist(), rotation=45, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left", title="NK-enriched clusters")

for i, val in enumerate(df["non_HDL"].values):
    ax.text(i, 1.02, f"{val:.0f}", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())

ax.text(0.0, 1.10, "Non-HDL", fontsize=9, fontweight="bold", transform=ax.transAxes)

png = figdir / "PanelI_GSE198339_NKcluster_composition.png"
pdf = figdir / "PanelI_GSE198339_NKcluster_composition.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(df.to_string(index=False))
