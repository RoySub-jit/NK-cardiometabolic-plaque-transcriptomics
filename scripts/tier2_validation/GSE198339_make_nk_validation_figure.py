from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE198339/extracted")
fig_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
fig_dir.mkdir(parents=True, exist_ok=True)

# Input files
nk_participant = pd.read_csv(base / "GSE198339_nk_validation_participant_level.csv")
nk_stats = pd.read_csv(base / "GSE198339_nk_validation_stats.csv")
prog_participant = pd.read_csv(base / "GSE198339_nk_program_participant_level.csv")
prog_stats = pd.read_csv(base / "GSE198339_nk_program_validation_stats.csv")

# -----------------------------
# Panel A: non-HDL vs NK cells resting
# -----------------------------
panelA_df = nk_participant.copy()
panelA_x = pd.to_numeric(panelA_df["non_HDL"], errors="coerce")
panelA_y = pd.to_numeric(panelA_df["NK cells resting"], errors="coerce")

a_row = nk_stats[nk_stats["outcome"] == "NK cells resting"].iloc[0]
a_rho = a_row["spearman_rho_non_HDL"]
a_p = a_row["spearman_p_non_HDL"]

# -----------------------------
# Panel B: non-HDL vs cytotoxic_core_score
# -----------------------------
panelB_df = prog_participant.copy()
panelB_x = pd.to_numeric(panelB_df["non_HDL"], errors="coerce")
panelB_y = pd.to_numeric(panelB_df["cytotoxic_core_score"], errors="coerce")

b_row = prog_stats[prog_stats["outcome"] == "cytotoxic_core_score"].iloc[0]
b_rho = b_row["spearman_rho_non_HDL"]
b_p = b_row["spearman_p_non_HDL"]

# -----------------------------
# Panel C: summary table
# -----------------------------
wanted = [
    ("NK cells resting", "NK resting proportion"),
    ("NK_combined_core", "NK core proportion"),
    ("NKG7", "NKG7 expression"),
    ("cytotoxic_core_score", "Cytotoxic core score"),
    ("gzmk_like_score", "GZMK-like score"),
]

rows = []
for outcome, label in wanted:
    if outcome in nk_stats["outcome"].values:
        row = nk_stats[nk_stats["outcome"] == outcome].iloc[0]
    else:
        row = prog_stats[prog_stats["outcome"] == outcome].iloc[0]

    rows.append([
        label,
        f"{row['spearman_rho_non_HDL']:.3f}",
        f"{row['spearman_p_non_HDL']:.4f}",
    ])

summary_df = pd.DataFrame(rows, columns=["Validation metric", "Spearman rho", "P value"])
summary_csv = base / "GSE198339_nk_validation_summary_table.csv"
summary_df.to_csv(summary_csv, index=False)

# -----------------------------
# Make 3-panel figure
# -----------------------------
fig = plt.figure(figsize=(14, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25])

# Panel A
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(panelA_x, panelA_y, s=38, alpha=0.85)
m1, b1 = np.polyfit(panelA_x, panelA_y, 1)
xx = np.linspace(panelA_x.min(), panelA_x.max(), 100)
ax1.plot(xx, m1 * xx + b1, linewidth=2)
ax1.set_xlabel("Non-HDL cholesterol")
ax1.set_ylabel("NK cells resting proportion")
ax1.set_title("A  NK abundance validation", loc="left", fontweight="bold")
ax1.text(
    0.03, 0.97,
    f"rho = {a_rho:.3f}\np = {a_p:.4f}",
    transform=ax1.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8")
)

# Panel B
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(panelB_x, panelB_y, s=38, alpha=0.85)
m2, b2 = np.polyfit(panelB_x, panelB_y, 1)
xx2 = np.linspace(panelB_x.min(), panelB_x.max(), 100)
ax2.plot(xx2, m2 * xx2 + b2, linewidth=2)
ax2.set_xlabel("Non-HDL cholesterol")
ax2.set_ylabel("Cytotoxic core score")
ax2.set_title("B  NK program validation", loc="left", fontweight="bold")
ax2.text(
    0.03, 0.97,
    f"rho = {b_rho:.3f}\np = {b_p:.4f}",
    transform=ax2.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8")
)

# Panel C
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
tbl = ax3.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    loc="center",
    cellLoc="center",
    colLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 1.6)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_linewidth(0.8)
    else:
        cell.set_linewidth(0.5)

ax3.set_title("C  Summary of external NK validation", loc="left", fontweight="bold", pad=12)

fig.suptitle("Tier 2A external PBMC validation of the NK lipid-associated axis", fontsize=14, fontweight="bold")
fig.tight_layout()

png_file = fig_dir / "Figure_tier2A_NK_validation.png"
pdf_file = fig_dir / "Figure_tier2A_NK_validation.pdf"
fig.savefig(png_file, dpi=300, bbox_inches="tight")
fig.savefig(pdf_file, bbox_inches="tight")
plt.close(fig)

print("Done.")
print(png_file)
print(pdf_file)
print(summary_csv)
