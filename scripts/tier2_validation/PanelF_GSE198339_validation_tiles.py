from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "tier2_validation" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

val_stats = pd.read_csv(
    base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_validation_stats.csv"
)
prog_stats = pd.read_csv(
    base / "data/tier2_validation/GSE198339/extracted/GSE198339_nk_program_validation_stats.csv"
)

rows = []

want_v = {
    "NK cells resting": "NK resting proportion",
}
for _, r in val_stats.iterrows():
    outcome = str(r["outcome"])
    if outcome in want_v:
        rows.append({
            "label": want_v[outcome],
            "rho": float(r["spearman_rho_non_HDL"]),
            "p": float(r["spearman_p_non_HDL"]),
        })

want_p = {
    "NKG7": "NKG7 expression",
    "cytotoxic_core_score": "Cytotoxic core score",
    "gzmk_like_score": "GZMK-like score",
}
for _, r in prog_stats.iterrows():
    outcome = str(r["outcome"])
    if outcome in want_p:
        rows.append({
            "label": want_p[outcome],
            "rho": float(r["spearman_rho_non_HDL"]),
            "p": float(r["spearman_p_non_HDL"]),
        })

df = pd.DataFrame(rows)
order = ["NKG7 expression", "NK resting proportion", "Cytotoxic core score", "GZMK-like score"]
df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
df = df.sort_values("label").reset_index(drop=True)

vals = df["rho"].values.reshape(-1, 1)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(4.8, 4.8))
im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0.75, vmax=0.87)

ax.set_title("F", loc="left", fontweight="bold", fontsize=18)
ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(df["label"])
ax.set_xticks([0])
ax.set_xticklabels(["Spearman ρ"])

for i in range(len(df)):
    ax.text(0, i, f"{df.loc[i,'rho']:.3f}\nP={df.loc[i,'p']:.4f}",
            ha="center", va="center", fontsize=9)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
cbar.set_label("Correlation strength")

png = figdir / "PanelF_GSE198339_validation_tiles.png"
pdf = figdir / "PanelF_GSE198339_validation_tiles.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
