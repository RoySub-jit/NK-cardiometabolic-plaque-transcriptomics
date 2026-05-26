
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
proc = base / "data/tier3_plaque_validation/GSE163154/processed"
figdir = base / "figures/tier3_plaque_validation/panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(proc / "GSE163154_nk_validation_sample_level.csv")
stats = pd.read_csv(proc / "GSE163154_nk_validation_stats.csv")

features = ["gzmk_like_score", "NKG7", "TYROBP", "GNLY", "KLRD1"]
features = [f for f in features if f in df.columns]

group_order = ["No_IPH", "IPH"]
fills = {"No_IPH": "#4C78A8", "IPH": "#E45756"}

nice = {
    "gzmk_like_score": "GZMK-like score",
    "NKG7": "NKG7 expression",
    "GNLY": "GNLY expression",
    "KLRD1": "KLRD1 expression",
    "TYROBP": "TYROBP expression"
}

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

n = len(features)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 4.0 * nrows))
axes = np.array(axes).reshape(-1)
rng = np.random.default_rng(42)

letters = list("KLMNO")

for ax, feature, letter in zip(axes, features, letters):
    x1 = pd.to_numeric(df.loc[df["group"] == "No_IPH", feature], errors="coerce").dropna().values
    x2 = pd.to_numeric(df.loc[df["group"] == "IPH", feature], errors="coerce").dropna().values

    parts = ax.violinplot(
        [x1, x2],
        positions=[1, 2],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for i, body in enumerate(parts["bodies"]):
        grp = group_order[i]
        body.set_facecolor(fills[grp])
        body.set_edgecolor("black")
        body.set_alpha(0.25)
        body.set_linewidth(0.8)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    for i, (vals, grp) in enumerate(zip([x1, x2], group_order), start=1):
        x = i + rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(x, vals, s=46, color=fills[grp], edgecolors="black", linewidths=0.5, zorder=3)

    st = stats.loc[stats["feature"] == feature].iloc[0]
    ptxt = f"{st['mannwhitney_p']:.2e}" if st["mannwhitney_p"] < 0.001 else f"{st['mannwhitney_p']:.4f}"
    ax.text(
        0.03, 0.97,
        f"P = {ptxt}\nΔ = {st['No_IPH_minus_IPH']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No IPH", "IPH"])
    ax.set_ylabel(nice[feature])
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes[len(features):]:
    ax.axis("off")

fig.suptitle("Independent plaque validation in GSE163154", fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout()

png = figdir / "Panel_GSE163154_second_plaque_validation_v2.png"
pdf = figdir / "Panel_GSE163154_second_plaque_validation_v2.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
