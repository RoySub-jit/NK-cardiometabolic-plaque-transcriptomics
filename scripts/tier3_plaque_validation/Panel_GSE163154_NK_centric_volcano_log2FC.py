
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
proc = base / "data/tier3_plaque_validation/GSE163154/processed"
figdir = base / "figures/tier3_plaque_validation/panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

de = pd.read_csv(proc / "GSE163154_bulk_gene_level_DE.csv").copy()

# true log2 fold change using group means
de["No_IPH_mean"] = pd.to_numeric(de["No_IPH_mean"], errors="coerce")
de["IPH_mean"] = pd.to_numeric(de["IPH_mean"], errors="coerce")
de["fdr"] = pd.to_numeric(de["fdr"], errors="coerce")

# small pseudocount to avoid division by zero
eps = 1e-6
de["log2FC_IPH_vs_No_IPH"] = np.log2((de["IPH_mean"] + eps) / (de["No_IPH_mean"] + eps))
de["neglog10_fdr"] = -np.log10(de["fdr"].clip(lower=1e-300))

nk_genes = {
    "GZMK","NKG7","GNLY","KLRD1","TYROBP","FCGR3A","PRF1","GZMB",
    "CCL5","IFITM3","B2M","CCL3","CCL4","IL32","TNF","NFKBIA",
    "IFI6","IFIT1","IFIT2","IFIT3","ISG15","MX1","OAS1","STAT1",
    "ITGAL","ITGB2","CCR7","CXCR3","ICAM3","CD74","HLA-A","HLA-B","HLA-DRA","TAP1","TAPBP","CTSW"
}

de["highlight"] = de["gene"].isin(nk_genes)

label_genes = [
    "GZMK","NKG7","GNLY","KLRD1","TYROBP","PRF1","GZMB","CCL5","IFITM3","B2M","FCGR3A",
    "CCL3","CCL4","IL32","IFI6","IFIT1","IFIT2","IFIT3","ISG15","MX1","OAS1","STAT1","CD74"
]
lab = de[de["gene"].isin(label_genes)].copy()

sig_cut = 0.05
fc_cut = 0.15

de["category"] = "Background"
de.loc[(de["highlight"]) & (de["fdr"] < sig_cut) & (de["log2FC_IPH_vs_No_IPH"] > fc_cut), "category"] = "IPH higher"
de.loc[(de["highlight"]) & (de["fdr"] < sig_cut) & (de["log2FC_IPH_vs_No_IPH"] < -fc_cut), "category"] = "No IPH higher"
de.loc[(de["highlight"]) & (de["category"] == "Background"), "category"] = "Highlighted"

color_map = {
    "Background": "#cfcfcf",
    "Highlighted": "#7f7f7f",
    "IPH higher": "#d55e5e",
    "No IPH higher": "#4c78a8",
}

size_map = {
    "Background": 10,
    "Highlighted": 20,
    "IPH higher": 34,
    "No IPH higher": 34,
}

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(8.2, 6.2))

for cat in ["Background", "Highlighted", "No IPH higher", "IPH higher"]:
    tmp = de[de["category"] == cat]
    if len(tmp) == 0:
        continue
    ax.scatter(
        tmp["log2FC_IPH_vs_No_IPH"],
        tmp["neglog10_fdr"],
        s=size_map[cat],
        color=color_map[cat],
        edgecolors="none" if cat == "Background" else "black",
        linewidths=0.35 if cat != "Background" else 0,
        alpha=0.85 if cat != "Background" else 0.55,
        zorder=2 if cat == "Background" else 3
    )

# thresholds
ax.axhline(-np.log10(sig_cut), linestyle="--", linewidth=1.0, color="0.4")
ax.axvline(fc_cut, linestyle="--", linewidth=1.0, color="0.4")
ax.axvline(-fc_cut, linestyle="--", linewidth=1.0, color="0.4")

# labels
for _, row in lab.iterrows():
    x = row["log2FC_IPH_vs_No_IPH"]
    y = row["neglog10_fdr"]
    if pd.isna(x) or pd.isna(y):
        continue
    txt_color = "#b22222" if x > 0 else "#1f4e79"
    ha = "left" if x >= 0 else "right"
    dx = 0.015 if x >= 0 else -0.015
    ax.text(
        x + dx, y, row["gene"],
        fontsize=8.5, color=txt_color, ha=ha, va="center"
    )

ax.set_title("GSE163154 NK-centric volcano", fontsize=13, fontweight="bold")
ax.set_xlabel("log2(mean IPH / mean No IPH)")
ax.set_ylabel("-log10(FDR)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#d55e5e", markeredgecolor="black", markersize=7, label="NK-related genes higher in IPH"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#4c78a8", markeredgecolor="black", markersize=7, label="NK-related genes higher in No IPH"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor="#7f7f7f", markeredgecolor="black", markersize=6, label="Highlighted NK / immune genes"),
]
ax.legend(handles=legend_elements, frameon=False, loc="upper left", fontsize=8.8)

png = figdir / "Panel_GSE163154_NK_centric_volcano_log2FC.png"
pdf = figdir / "Panel_GSE163154_NK_centric_volcano_log2FC.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(de.loc[de['highlight'], ['gene','log2FC_IPH_vs_No_IPH','fdr','category']].sort_values('fdr').to_string(index=False))
