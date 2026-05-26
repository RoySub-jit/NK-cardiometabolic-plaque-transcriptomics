from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
tier2_base = project_dir / "data" / "tier2_validation" / "GSE198339" / "extracted"
fig_dir = project_dir / "figures" / "tier2_validation"
fig_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load discovery and validation inputs
# -----------------------------
allen_nk = pd.read_csv(results_dir / "nk_primary_nonhdl_model_input.csv")
allen_summary = pd.read_csv(results_dir / "nk_primary_nonhdl_model_summary.csv")

ext_nk = pd.read_csv(tier2_base / "GSE198339_nk_validation_participant_level.csv")
ext_nk_stats = pd.read_csv(tier2_base / "GSE198339_nk_validation_stats.csv")

prog_stats = pd.read_csv(tier2_base / "GSE198339_nk_program_validation_stats.csv")

# -----------------------------
# Allen discovery stats
# -----------------------------
allen_beta = float(allen_summary.loc[0, "beta"])
allen_p = float(allen_summary.loc[0, "p_value"])
allen_r2 = float(allen_summary.loc[0, "r_squared"])

# -----------------------------
# External validation stats
# -----------------------------
nk_rest_row = ext_nk_stats[ext_nk_stats["outcome"] == "NK cells resting"].iloc[0]
ext_rho = float(nk_rest_row["spearman_rho_non_HDL"])
ext_p = float(nk_rest_row["spearman_p_non_HDL"])

# -----------------------------
# Program heatmap values
# -----------------------------
wanted = [
    "GZMK",
    "NKG7",
    "GNLY",
    "PRF1",
    "GZMB",
    "FCGR3A",
    "KLRD1",
    "TYROBP",
    "cytotoxic_core_score",
    "gzmk_like_score",
]
prog_sub = prog_stats[prog_stats["outcome"].isin(wanted)].copy()
prog_sub["label"] = prog_sub["outcome"].replace({
    "cytotoxic_core_score": "cytotoxic_core_score",
    "gzmk_like_score": "gzmk_like_score",
})
prog_sub = prog_sub.set_index("label").loc[wanted].reset_index()

# -----------------------------
# Make figure
# -----------------------------
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.35, wspace=0.3)

# Panel A: Allen discovery scatter
ax1 = fig.add_subplot(gs[0, 0])
x1 = pd.to_numeric(allen_nk["lip.cholesterol_non_hdl"], errors="coerce")
y1 = pd.to_numeric(allen_nk["proportion"], errors="coerce")
mask1 = x1.notna() & y1.notna()
x1 = x1[mask1]
y1 = y1[mask1]

ax1.scatter(x1, y1, s=28, alpha=0.8)
m1, b1 = np.polyfit(x1, y1, 1)
xx1 = np.linspace(x1.min(), x1.max(), 100)
ax1.plot(xx1, m1 * xx1 + b1, linewidth=2)

ax1.set_xlabel("Non-HDL cholesterol")
ax1.set_ylabel("GZMK+ CD56dim NK proportion")
ax1.set_title("A  Allen discovery cohort", loc="left", fontweight="bold")
ax1.text(
    0.03, 0.97,
    f"β = {allen_beta:.6f}\nP = {allen_p:.4f}\nR² = {allen_r2:.4f}",
    transform=ax1.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8")
)

# Panel B: external validation scatter
ax2 = fig.add_subplot(gs[0, 1])
x2 = pd.to_numeric(ext_nk["non_HDL"], errors="coerce")
y2 = pd.to_numeric(ext_nk["NK cells resting"], errors="coerce")
mask2 = x2.notna() & y2.notna()
x2 = x2[mask2]
y2 = y2[mask2]

ax2.scatter(x2, y2, s=38, alpha=0.85)
m2, b2 = np.polyfit(x2, y2, 1)
xx2 = np.linspace(x2.min(), x2.max(), 100)
ax2.plot(xx2, m2 * xx2 + b2, linewidth=2)

ax2.set_xlabel("Non-HDL cholesterol")
ax2.set_ylabel("NK cells resting proportion")
ax2.set_title("B  External PBMC validation cohort", loc="left", fontweight="bold")
ax2.text(
    0.03, 0.97,
    f"Spearman ρ = {ext_rho:.3f}\nP = {ext_p:.4f}",
    transform=ax2.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8")
)

# Panel C: program-level heatmap
ax3 = fig.add_subplot(gs[1, 0])
heat_vals = prog_sub["spearman_rho_non_HDL"].to_numpy(dtype=float).reshape(-1, 1)
im = ax3.imshow(heat_vals, aspect="auto", vmin=-1, vmax=1)

ax3.set_yticks(np.arange(len(prog_sub)))
ax3.set_yticklabels(prog_sub["label"])
ax3.set_xticks([0])
ax3.set_xticklabels(["Non-HDL"])
ax3.set_title("C  NK program-level validation", loc="left", fontweight="bold")

for i, (_, row) in enumerate(prog_sub.iterrows()):
    rho = row["spearman_rho_non_HDL"]
    p = row["spearman_p_non_HDL"]
    ax3.text(
        0, i,
        f"ρ={rho:.2f}\nP={p:.3f}",
        ha="center", va="center",
        fontsize=8,
        color="black" if abs(rho) < 0.65 else "white"
    )

cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Spearman rho")

# Panel D: pathway leads table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
ax4.set_title("D  Candidate cardiometabolic NK pathway leads", loc="left", fontweight="bold", pad=10)

pathway_rows = [
    ["Cytotoxic effector program", "NKG7, GNLY, PRF1, GZMB", "Positive non-HDL correlation in external NK cells"],
    ["NK activation / Fc receptor signaling", "FCGR3A, TYROBP, KLRD1", "Supports activated NK-like lipid axis"],
    ["GZMK-like remodeling", "GZMK, NKG7, KLRD1", "Concordant with Allen GZMK+ CD56dim NK discovery"],
]

tbl = ax4.table(
    cellText=pathway_rows,
    colLabels=["Candidate pathway lead", "Genes", "Interpretive note"],
    loc="center",
    cellLoc="left",
    colLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.15, 1.6)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_linewidth(0.8)
    else:
        cell.set_linewidth(0.5)

fig.suptitle(
    "Cross-dataset concordance of the lipid-associated NK axis and candidate pathway leads",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
fig.tight_layout()

png_file = fig_dir / "Figure_tier2A_NK_concordance_pathway.png"
pdf_file = fig_dir / "Figure_tier2A_NK_concordance_pathway.pdf"
fig.savefig(png_file, dpi=300, bbox_inches="tight")
fig.savefig(pdf_file, bbox_inches="tight")
plt.close(fig)

print("Done.")
print(png_file)
print(pdf_file)
