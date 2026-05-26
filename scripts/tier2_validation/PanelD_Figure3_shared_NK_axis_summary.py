
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
outdir = base / "results" / "tier2_validation"
datadir = base / "data/tier2_validation/GSE198339/extracted"
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

# PBMC stats
pbmc_val = pd.read_csv(datadir / "GSE198339_nk_validation_stats.csv")
pbmc_prog = pd.read_csv(datadir / "GSE198339_nk_program_validation_stats.csv")

# Plaque stats
plaque_main = pd.read_csv(outdir / "GSE224273_nk_high_targeted_stats.csv")
plaque_gene = pd.read_csv(outdir / "GSE224273_nk_high_core_gene_stats.csv")

rows = []

# PBMC readouts
map_pbmc = {
    "NKG7": "NKG7 expression",
    "gzmk_like_score": "GZMK-like score",
    "NK cells resting": "NK resting proportion",
}
for _, r in pbmc_prog.iterrows():
    if str(r["outcome"]) in ["NKG7", "gzmk_like_score"]:
        rows.append({
            "readout": map_pbmc[str(r["outcome"])],
            "dataset": "PBMC",
            "effect": float(r["spearman_rho_non_HDL"]),
            "p_value": float(r["spearman_p_non_HDL"]),
        })
for _, r in pbmc_val.iterrows():
    if str(r["outcome"]) == "NK cells resting":
        rows.append({
            "readout": map_pbmc[str(r["outcome"])],
            "dataset": "PBMC",
            "effect": float(r["spearman_rho_non_HDL"]),
            "p_value": float(r["spearman_p_non_HDL"]),
        })

# Plaque readouts
for _, r in plaque_main.iterrows():
    if str(r["endpoint"]) == "gzmk_like_score_mean":
        rows.append({
            "readout": "GZMK-like score",
            "dataset": "Plaque",
            "effect": float(r["asym_minus_symptomatic_mean"]),
            "p_value": float(r["exact_permutation_p"]),
        })

for _, r in plaque_gene.iterrows():
    g = str(r["gene"])
    if g in ["NKG7", "KLRD1", "GNLY"]:
        rows.append({
            "readout": {"NKG7":"NKG7 expression","KLRD1":"KLRD1 expression","GNLY":"GNLY expression"}[g],
            "dataset": "Plaque",
            "effect": float(r["asym_minus_sym_mean"]),
            "p_value": float(r["exact_permutation_p"]),
        })

df = pd.DataFrame(rows)

order = ["NKG7 expression", "NK resting proportion", "GZMK-like score", "GNLY expression", "KLRD1 expression"]
mat = df.pivot(index="readout", columns="dataset", values="effect").reindex(order)
piv_p = df.pivot(index="readout", columns="dataset", values="p_value").reindex(order)

# z-scale per column for display
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

fig, ax = plt.subplots(figsize=(4.8, 5.6))
im = ax.imshow(mat_z.values, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)

ax.set_title("D", loc="left", fontweight="bold", fontsize=18)
ax.set_xticks(np.arange(len(mat_z.columns)))
ax.set_xticklabels(mat_z.columns.tolist())
ax.set_yticks(np.arange(len(mat_z.index)))
ax.set_yticklabels(mat_z.index.tolist())

for i, readout in enumerate(mat.index):
    for j, ds in enumerate(mat.columns):
        val = mat.loc[readout, ds]
        p = piv_p.loc[readout, ds]
        if pd.isna(val):
            ax.text(j, i, "NA", ha="center", va="center", fontsize=8)
        else:
            star = "*" if pd.notna(p) and p < 0.05 else ""
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center", fontsize=8.3)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("Relative effect strength")

png = figdir / "PanelD_Figure3_shared_NK_axis_summary.png"
pdf = figdir / "PanelD_Figure3_shared_NK_axis_summary.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(mat)
print(piv_p)
