from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

rob = pd.read_csv(base / "results/first_dataset/primary_robustness_checks_combined_summary.csv")
age_row = rob[rob["analysis"] == "NK_age_continuous"].iloc[0]
loo_row = rob[rob["analysis"] == "NK_leave_one_out"].iloc[0]

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

fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 0.66], hspace=0.03)

# top stats band
ax_top = fig.add_subplot(gs[0])
ax_top.axis("off")
ax_top.text(0.0, 0.92, "E", fontsize=18, fontweight="bold", ha="left", va="top")
stat_text = (
    f"Adjusted model: β = {age_row['beta']:.6f}, P = {age_row['p_value']:.4f}, "
    f"95% CI [{age_row['conf_low']:.6f}, {age_row['conf_high']:.6f}]\n"
    f"Leave-one-out: median β = {loo_row['beta_median']:.6f}, "
    f"median P = {loo_row['p_median']:.4f}"
)
ax_top.text(
    0.08, 0.62, stat_text,
    fontsize=10, ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95)
)

# main plot
ax = fig.add_subplot(gs[1])

ax.errorbar(
    [age_row["beta"]], [1],
    xerr=[[age_row["beta"] - age_row["conf_low"]], [age_row["conf_high"] - age_row["beta"]]],
    fmt='o', color='black', capsize=4, markersize=7
)

ax.plot([loo_row["beta_min"], loo_row["beta_max"]], [0, 0], color="black", linewidth=1.6)
ax.scatter([loo_row["beta_median"]], [0], color="black", s=60, zorder=3)

ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0)
ax.set_yticks([1, 0])
ax.set_yticklabels(["Adjusted model", "Leave-one-out"])
ax.set_xlabel("Effect size")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

png = figdir / "PanelE_NK_model_leaveoneout.png"
pdf = figdir / "PanelE_NK_model_leaveoneout.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
