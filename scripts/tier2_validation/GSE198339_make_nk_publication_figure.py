from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -----------------------------
# Paths
# -----------------------------
project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
tier2_base = project_dir / "data" / "tier2_validation" / "GSE198339" / "extracted"
fig_dir = project_dir / "figures" / "tier2_validation"
fig_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
allen_input = pd.read_csv(results_dir / "nk_primary_nonhdl_model_input.csv")
allen_summary = pd.read_csv(results_dir / "nk_primary_nonhdl_model_summary.csv")

ext_nk = pd.read_csv(tier2_base / "GSE198339_nk_validation_participant_level.csv")
ext_nk_stats = pd.read_csv(tier2_base / "GSE198339_nk_validation_stats.csv")

prog_stats = pd.read_csv(tier2_base / "GSE198339_nk_program_validation_stats.csv")

# -----------------------------
# Basic style
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# -----------------------------
# Discovery panel
# -----------------------------
allen_x = pd.to_numeric(allen_input["lip.cholesterol_non_hdl"], errors="coerce")
allen_y = pd.to_numeric(allen_input["proportion"], errors="coerce")
mask = allen_x.notna() & allen_y.notna()
allen_x = allen_x[mask]
allen_y = allen_y[mask]

allen_beta = float(allen_summary.loc[0, "beta"])
allen_p = float(allen_summary.loc[0, "p_value"])

# -----------------------------
# External validation panel
# -----------------------------
ext_x = pd.to_numeric(ext_nk["non_HDL"], errors="coerce")
ext_y = pd.to_numeric(ext_nk["NK cells resting"], errors="coerce")
mask2 = ext_x.notna() & ext_y.notna()
ext_x = ext_x[mask2]
ext_y = ext_y[mask2]

ext_row = ext_nk_stats[ext_nk_stats["outcome"] == "NK cells resting"].iloc[0]
ext_rho = float(ext_row["spearman_rho_non_HDL"])
ext_p = float(ext_row["spearman_p_non_HDL"])

# -----------------------------
# Program validation heatmap
# -----------------------------
wanted = [
    "NKG7",
    "GNLY",
    "PRF1",
    "GZMB",
    "FCGR3A",
    "TYROBP",
    "KLRD1",
    "GZMK",
    "cytotoxic_core_score",
    "gzmk_like_score",
]
program_labels = {
    "NKG7": "NKG7",
    "GNLY": "GNLY",
    "PRF1": "PRF1",
    "GZMB": "GZMB",
    "FCGR3A": "FCGR3A",
    "TYROBP": "TYROBP",
    "KLRD1": "KLRD1",
    "GZMK": "GZMK",
    "cytotoxic_core_score": "cytotoxic core",
    "gzmk_like_score": "GZMK-like",
}
prog_sub = prog_stats[prog_stats["outcome"].isin(wanted)].copy()
prog_sub = prog_sub.set_index("outcome").loc[wanted].reset_index()
prog_sub["label"] = prog_sub["outcome"].map(program_labels)

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(13.5, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.28)

# Panel A
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(allen_x, allen_y, s=28, alpha=0.85)
m1, b1 = np.polyfit(allen_x, allen_y, 1)
xx1 = np.linspace(allen_x.min(), allen_x.max(), 100)
ax1.plot(xx1, m1 * xx1 + b1, linewidth=2)
ax1.set_xlabel("Non-HDL cholesterol")
ax1.set_ylabel("GZMK+ CD56dim NK proportion")
ax1.set_title("A  Allen discovery cohort", loc="left", fontweight="bold")
ax1.text(
    0.03, 0.97,
    f"β = {allen_beta:.6f}\nP = {allen_p:.4f}",
    transform=ax1.transAxes,
    va="top", ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75")
)

# Panel B
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(ext_x, ext_y, s=36, alpha=0.9)
m2, b2 = np.polyfit(ext_x, ext_y, 1)
xx2 = np.linspace(ext_x.min(), ext_x.max(), 100)
ax2.plot(xx2, m2 * xx2 + b2, linewidth=2)
ax2.set_xlabel("Non-HDL cholesterol")
ax2.set_ylabel("NK resting proportion")
ax2.set_title("B  External PBMC validation cohort", loc="left", fontweight="bold")
ax2.text(
    0.03, 0.97,
    f"ρ = {ext_rho:.3f}\nP = {ext_p:.4f}",
    transform=ax2.transAxes,
    va="top", ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75")
)

# Panel C
ax3 = fig.add_subplot(gs[1, 0])
heat_vals = prog_sub["spearman_rho_non_HDL"].to_numpy(dtype=float).reshape(-1, 1)
im = ax3.imshow(heat_vals, aspect="auto", vmin=-1, vmax=1)
ax3.set_yticks(np.arange(len(prog_sub)))
ax3.set_yticklabels(prog_sub["label"])
ax3.set_xticks([0])
ax3.set_xticklabels(["Non-HDL"])
ax3.set_title("C  External NK program validation", loc="left", fontweight="bold")

for i, (_, row) in enumerate(prog_sub.iterrows()):
    rho = row["spearman_rho_non_HDL"]
    p = row["spearman_p_non_HDL"]
    txt_color = "white" if abs(rho) >= 0.65 else "black"
    ax3.text(0, i, f"ρ={rho:.2f}\nP={p:.3f}", ha="center", va="center", fontsize=8, color=txt_color)

cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Spearman rho")

# Panel D
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
ax4.set_title("D  Candidate NK cardiometabolic modules", loc="left", fontweight="bold", pad=8)

boxes = [
    {
        "xy": (0.03, 0.68), "w": 0.92, "h": 0.23,
        "title": "Cytotoxic effector",
        "genes": "NKG7 · GNLY · PRF1 · GZMB",
        "path": "vascular cell injury / plaque inflammation"
    },
    {
        "xy": (0.03, 0.39), "w": 0.92, "h": 0.23,
        "title": "NK activation / signaling",
        "genes": "FCGR3A · TYROBP · KLRD1",
        "path": "innate activation / Fc receptor signaling"
    },
    {
        "xy": (0.03, 0.10), "w": 0.92, "h": 0.23,
        "title": "GZMK-like inflammatory remodeling",
        "genes": "GZMK · NKG7 · KLRD1",
        "path": "chronic inflammatory NK remodeling"
    },
]

for box in boxes:
    patch = FancyBboxPatch(
        box["xy"], box["w"], box["h"],
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0, edgecolor="0.5", facecolor="white",
        transform=ax4.transAxes
    )
    ax4.add_patch(patch)
    ax4.text(box["xy"][0] + 0.02, box["xy"][1] + 0.16, box["title"],
             transform=ax4.transAxes, fontsize=10, fontweight="bold", va="center")
    ax4.text(box["xy"][0] + 0.02, box["xy"][1] + 0.09, box["genes"],
             transform=ax4.transAxes, fontsize=9.5, va="center")
    ax4.text(box["xy"][0] + 0.02, box["xy"][1] + 0.03, box["path"],
             transform=ax4.transAxes, fontsize=8.8, style="italic", va="center")

fig.suptitle(
    "Cross-dataset concordance of the lipid-associated NK axis and candidate NK modules",
    fontsize=15, fontweight="bold", y=0.98
)
fig.tight_layout()

png_file = fig_dir / "Figure_tier2A_NK_publication_style.png"
pdf_file = fig_dir / "Figure_tier2A_NK_publication_style.pdf"
fig.savefig(png_file, dpi=300, bbox_inches="tight")
fig.savefig(pdf_file, bbox_inches="tight")
plt.close(fig)

print("Done.")
print(png_file)
print(pdf_file)
