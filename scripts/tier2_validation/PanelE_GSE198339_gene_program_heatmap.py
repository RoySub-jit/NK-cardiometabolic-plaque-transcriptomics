from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_program_participant_level.csv"
).copy()

genes = ["NKG7", "GZMK", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRD1", "TYROBP"]
genes = [g for g in genes if g in df.columns]

df["non_HDL"] = pd.to_numeric(df["non_HDL"], errors="coerce")
for g in genes:
    df[g] = pd.to_numeric(df[g], errors="coerce")

df = df.dropna(subset=["non_HDL"] + genes).copy()
df = df.sort_values("non_HDL").reset_index(drop=True)

mat = df[genes].copy()
mat = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8.6, 4.6))
im = ax.imshow(mat.T.values, aspect="auto", cmap="Blues", vmin=-1.5, vmax=1.5)

ax.set_title("E", loc="left", fontweight="bold", fontsize=18)
ax.set_yticks(np.arange(len(genes)))
ax.set_yticklabels(genes)
ax.set_xticks([])
ax.set_xlabel("Participants ordered by non-HDL")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Relative expression")

png = figdir / "PanelE_GSE198339_gene_program_heatmap.png"
pdf = figdir / "PanelE_GSE198339_gene_program_heatmap.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
