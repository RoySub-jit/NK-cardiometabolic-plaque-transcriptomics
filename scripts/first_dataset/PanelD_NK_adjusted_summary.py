from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

rob = pd.read_csv(base / "results/first_dataset/primary_robustness_checks_combined_summary.csv")
row = rob[rob["analysis"] == "NK_age_continuous"].iloc[0]

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

fig = plt.figure(figsize=(6.8, 4.8))
gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 0.66], hspace=0.03)

ax_top = fig.add_subplot(gs[0])
ax_top.axis("off")
ax_top.text(0.0, 0.92, "D", fontsize=18, fontweight="bold", ha="left", va="top")
ax_top.text(
    0.08, 0.62,
    "Adjusted association of non-HDL cholesterol with\nGZMK+ CD56dim NK proportion",
    fontsize=10, ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95)
)

ax = fig.add_subplot(gs[1])

beta = row["beta"]
ci_low = row["conf_low"]
ci_high = row["conf_high"]
pval = row["p_value"]

ax.errorbar(
    [beta], [0],
    xerr=[[beta - ci_low], [ci_high - beta]],
    fmt='o', color='black', capsize=4, markersize=7
)
ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0)

ax.set_yticks([0])
ax.set_yticklabels(["Adjusted model"])
ax.set_xlabel("Effect size")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(
    0.03, 0.97,
    f"β = {beta:.6f}\nP = {pval:.4f}\n95% CI [{ci_low:.6f}, {ci_high:.6f}]",
    transform=ax.transAxes, ha="left", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelD_NK_adjusted_summary.png"
pdf = figdir / "PanelD_NK_adjusted_summary.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
