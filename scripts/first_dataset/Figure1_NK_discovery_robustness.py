from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import linregress

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset"
figdir.mkdir(parents=True, exist_ok=True)

nk = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv")
rob = pd.read_csv(base / "results/first_dataset/primary_robustness_checks_combined_summary.csv")

sub = nk[nk["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()
sub["lip.cholesterol_non_hdl"] = pd.to_numeric(sub["lip.cholesterol_non_hdl"], errors="coerce")
sub["proportion"] = pd.to_numeric(sub["proportion"], errors="coerce")
sub = sub.dropna(subset=["lip.cholesterol_non_hdl", "proportion"]).copy()

sub["nonhdl_tertile"] = pd.qcut(sub["lip.cholesterol_non_hdl"], q=3, labels=["Low", "Mid", "High"])

lr = linregress(sub["lip.cholesterol_non_hdl"], sub["proportion"])
xline = np.linspace(sub["lip.cholesterol_non_hdl"].min(), sub["lip.cholesterol_non_hdl"].max(), 200)
yline = lr.intercept + lr.slope * xline

age_row = rob[rob["analysis"] == "NK_age_continuous"].iloc[0]
loo_row = rob[rob["analysis"] == "NK_leave_one_out"].iloc[0]
boot_row = rob[rob["analysis"] == "NK_bootstrap"].iloc[0]

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(13.4, 8.6))
gs = fig.add_gridspec(3, 4, height_ratios=[0.95, 1.1, 1.05], hspace=0.7, wspace=0.65)

# A schematic
axA = fig.add_subplot(gs[0, :])
axA.axis("off")
axA.set_title("A", loc="left", fontweight="bold", pad=2)

boxes = [
    ((0.03, 0.24), 0.22, 0.48, "Allen healthy adult cohort", "74 blood samples\nscRNA-seq + clinical labs"),
    ((0.30, 0.24), 0.22, 0.48, "NK state resolution", "L3 NK subtypes\nincluding GZMK+ CD56dim NK"),
    ((0.57, 0.24), 0.18, 0.48, "Primary test", "non-HDL ~ NK proportion\n+ age + sex + CMV + BMI"),
    ((0.80, 0.24), 0.17, 0.48, "Output", "lipid-associated\nGZMK-like NK axis"),
]

for (xy, w, h, title, txt) in boxes:
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0, edgecolor="0.45", facecolor="white",
        transform=axA.transAxes
    )
    axA.add_patch(patch)
    axA.text(xy[0] + 0.015, xy[1] + 0.31, title, transform=axA.transAxes,
             fontsize=10, fontweight="bold", va="center")
    axA.text(xy[0] + 0.015, xy[1] + 0.14, txt, transform=axA.transAxes,
             fontsize=9, va="center")

for x1, x2 in [(0.25, 0.30), (0.52, 0.57), (0.75, 0.80)]:
    axA.annotate("", xy=(x2, 0.48), xytext=(x1, 0.48),
                 xycoords=axA.transAxes, textcoords=axA.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.2))

# B scatter
axB = fig.add_subplot(gs[1, 0:2])
axB.set_title("B", loc="left", fontweight="bold")
axB.scatter(sub["lip.cholesterol_non_hdl"], sub["proportion"], s=42, edgecolors="black", linewidths=0.6)
axB.plot(xline, yline, linewidth=1.4)
axB.set_xlabel("Non-HDL cholesterol")
axB.set_ylabel("GZMK+ CD56dim NK proportion")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.text(
    0.03, 0.97,
    f"n = {len(sub)}\nSlope = {lr.slope:.4f}\nP = {lr.pvalue:.4f}\nR² = {lr.rvalue**2:.3f}",
    transform=axB.transAxes, ha="left", va="top", fontsize=8.8,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

# C tertiles
axC = fig.add_subplot(gs[1, 2:4])
axC.set_title("C", loc="left", fontweight="bold")
groups = [sub.loc[sub["nonhdl_tertile"] == lab, "proportion"].values for lab in ["Low", "Mid", "High"]]
bp = axC.boxplot(
    groups, positions=[1, 2, 3], widths=0.55, patch_artist=True, showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    boxprops=dict(linewidth=1.0), whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0)
)
fills = ["#dbe9f6", "#9ecae1", "#4C78A8"]
for patch, fc in zip(bp["boxes"], fills):
    patch.set_facecolor(fc)
    patch.set_edgecolor("black")

rng = np.random.default_rng(42)
for i, vals in enumerate(groups, start=1):
    x = i + rng.uniform(-0.08, 0.08, size=len(vals))
    axC.scatter(x, vals, s=34, edgecolors="black", linewidths=0.5, zorder=3)

axC.set_xticks([1, 2, 3])
axC.set_xticklabels(["Low", "Mid", "High"])
axC.set_ylabel("GZMK+ CD56dim NK proportion")
axC.set_xlabel("Non-HDL tertile")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)

# D primary model
axD = fig.add_subplot(gs[2, 0])
axD.set_title("D", loc="left", fontweight="bold")
axD.errorbar(
    [age_row["beta"]], [0],
    xerr=[[age_row["beta"] - age_row["conf_low"]], [age_row["conf_high"] - age_row["beta"]]],
    fmt='o', color='black', capsize=4
)
axD.axvline(0, color="0.35", linestyle=":", linewidth=1.0)
axD.set_yticks([0])
axD.set_yticklabels(["Primary model"])
axD.set_xlabel("Effect size")
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.text(
    0.03, 0.97,
    f"β = {age_row['beta']:.6f}\nP = {age_row['p_value']:.4f}\n95% CI [{age_row['conf_low']:.6f}, {age_row['conf_high']:.6f}]",
    transform=axD.transAxes, ha="left", va="top", fontsize=8.5
)

# E leave-one-out
axE = fig.add_subplot(gs[2, 1])
axE.set_title("E", loc="left", fontweight="bold")
vals = [loo_row["beta_min"], loo_row["beta_median"], loo_row["beta_max"]]
axE.plot([1, 1], [vals[0], vals[2]], color="black", linewidth=1.5)
axE.scatter([1, 1, 1], vals, color="black", s=[30, 60, 30], zorder=3)
axE.axhline(0, color="0.35", linestyle=":", linewidth=1.0)
axE.set_xlim(0.6, 1.4)
axE.set_xticks([1])
axE.set_xticklabels(["Leave-one-out"])
axE.set_ylabel("Effect size")
axE.spines["top"].set_visible(False)
axE.spines["right"].set_visible(False)
axE.text(
    0.03, 0.97,
    f"n models = {int(loo_row['n_leave_one_out_models'])}\nMedian β = {loo_row['beta_median']:.6f}\nMedian P = {loo_row['p_median']:.4f}",
    transform=axE.transAxes, ha="left", va="top", fontsize=8.5
)

# F bootstrap
axF = fig.add_subplot(gs[2, 2:4])
axF.set_title("F", loc="left", fontweight="bold")
boot_vals = [boot_row["beta_ci_low_2_5"], boot_row["beta_median"], boot_row["beta_ci_high_97_5"]]
axF.plot([1, 1], [boot_vals[0], boot_vals[2]], color="black", linewidth=1.7)
axF.scatter([1], [boot_vals[1]], color="black", s=70, zorder=3)
axF.axhline(0, color="0.35", linestyle=":", linewidth=1.0)
axF.set_xlim(0.7, 1.3)
axF.set_xticks([1])
axF.set_xticklabels(["Bootstrap"])
axF.set_ylabel("Effect size")
axF.spines["top"].set_visible(False)
axF.spines["right"].set_visible(False)
axF.text(
    0.03, 0.97,
    f"n = {int(boot_row['n_bootstrap'])}\nMedian β = {boot_row['beta_median']:.6f}\n95% CI [{boot_row['beta_ci_low_2_5']:.6f}, {boot_row['beta_ci_high_97_5']:.6f}]\nExpected sign = {boot_row['prop_expected_sign']:.3f}",
    transform=axF.transAxes, ha="left", va="top", fontsize=8.5
)

fig.suptitle(
    "Figure 1. Discovery and robustness of a lipid-associated GZMK-like NK-cell axis in healthy adult blood",
    fontsize=15, fontweight="bold", y=0.99
)

fig.tight_layout()
png = figdir / "Figure1_NK_discovery_robustness.png"
pdf = figdir / "Figure1_NK_discovery_robustness.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
