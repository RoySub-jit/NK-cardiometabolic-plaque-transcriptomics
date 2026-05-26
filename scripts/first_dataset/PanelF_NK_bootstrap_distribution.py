from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

boot = pd.read_csv(base / "results/first_dataset/nk_primary_nonhdl_bootstrap_details.csv")
rob = pd.read_csv(base / "results/first_dataset/primary_robustness_checks_combined_summary.csv")
boot_row = rob[rob["analysis"] == "NK_bootstrap"].iloc[0]

boot["beta"] = pd.to_numeric(boot["beta"], errors="coerce")
boot = boot.dropna(subset=["beta"]).copy()

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

fig = plt.figure(figsize=(6.8, 5.8))
gs = fig.add_gridspec(2, 1, height_ratios=[0.30, 0.70], hspace=0.03)

# top stats band
ax_top = fig.add_subplot(gs[0])
ax_top.axis("off")
ax_top.text(0.0, 0.92, "F", fontsize=18, fontweight="bold", ha="left", va="top")
stat_text = (
    f"n = {int(boot_row['n_bootstrap'])}\n"
    f"Median β = {boot_row['beta_median']:.6f}\n"
    f"95% CI [{boot_row['beta_ci_low_2_5']:.6f}, {boot_row['beta_ci_high_97_5']:.6f}]\n"
    f"Expected sign = {boot_row['prop_expected_sign']:.3f}"
)
ax_top.text(
    0.08, 0.62, stat_text,
    fontsize=10, ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95)
)

# histogram
ax = fig.add_subplot(gs[1])

ax.hist(boot["beta"], bins=28, color="#4a90d9", edgecolor="white")
ax.axvline(boot_row["beta_median"], color="black", linewidth=1.6)
ax.axvline(boot_row["beta_ci_low_2_5"], color="black", linestyle="--", linewidth=1.0)
ax.axvline(boot_row["beta_ci_high_97_5"], color="black", linestyle="--", linewidth=1.0)
ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0)

ax.set_xlabel("Bootstrapped effect size")
ax.set_ylabel("Count")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

png = figdir / "PanelF_NK_bootstrap_distribution.png"
pdf = figdir / "PanelF_NK_bootstrap_distribution.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
