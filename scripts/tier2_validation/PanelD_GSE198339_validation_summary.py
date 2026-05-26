from pathlib import Path
import pandas as pd
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

# validation file
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

# program file
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

order = [
    "NKG7 expression",
    "NK resting proportion",
    "Cytotoxic core score",
    "GZMK-like score",
]
df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
df = df.sort_values("label").reset_index(drop=True)

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

fig, ax = plt.subplots(figsize=(7.2, 5.4))

y = list(range(len(df)))[::-1]
ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0)

for yi, (_, row) in zip(y, df.iterrows()):
    ax.scatter(row["rho"], yi, s=70, color="#2f78b7", edgecolors="black", linewidths=0.5, zorder=3)
    ax.text(0.905, yi, f"P = {row['p']:.4f}", va="center", ha="right", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(df["label"].tolist())
ax.set_xlim(0, 0.95)
ax.set_xlabel("Spearman correlation with non-HDL")
ax.set_title("D", loc="left", fontweight="bold", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

png = figdir / "PanelD_GSE198339_validation_summary.png"
pdf = figdir / "PanelD_GSE198339_validation_summary.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print(df.to_string(index=False))
print("Saved:", png)
print("Saved:", pdf)
