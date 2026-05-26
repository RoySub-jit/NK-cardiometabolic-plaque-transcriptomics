from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
figdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/figures/tier2_validation")
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(outdir / "GSE253902_dc_presence_sample_scores_fixed.csv")

# Sort: asymptomatic first, then symptomatic by descending cDC2-like score
df["status_order"] = df["status"].map({"Asymptomatic": 0, "Symptomatic": 1})
df = df.sort_values(["status_order", "cdc2_score_mean"], ascending=[True, False]).reset_index(drop=True)

summary_df = df[[
    "sample", "status", "n_cells",
    "cdc2_score_mean", "pdc_score_mean", "pdc_minus_cdc2_mean"
]].copy()

summary_df["n_cells"] = summary_df["n_cells"].astype(int)
for col in ["cdc2_score_mean", "pdc_score_mean", "pdc_minus_cdc2_mean"]:
    summary_df[col] = summary_df[col].map(lambda x: f"{x:.3f}")

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig = plt.figure(figsize=(13.5, 7.8))
gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], width_ratios=[1.0, 1.1], hspace=0.38, wspace=0.32)

# Panel A: workflow schematic
ax1 = fig.add_subplot(gs[0, :])
ax1.axis("off")
ax1.set_title("A  DC plaque translation workflow", loc="left", fontweight="bold", pad=8)

boxes = [
    {
        "xy": (0.02, 0.25), "w": 0.22, "h": 0.5,
        "title": "Allen discovery",
        "text": "Healthy adult blood atlas\nhs-CRP associated with lower\nHLA-DRhi cDC2 abundance"
    },
    {
        "xy": (0.29, 0.25), "w": 0.22, "h": 0.5,
        "title": "Public carotid plaque atlas",
        "text": "GSE253902 CITE-seq/GEX\n6 plaque samples\n5 symptomatic, 1 asymptomatic"
    },
    {
        "xy": (0.56, 0.25), "w": 0.18, "h": 0.5,
        "title": "DC/APC mapping",
        "text": "Score cDC2-like\nand pDC-like\nprograms per cell"
    },
    {
        "xy": (0.79, 0.25), "w": 0.18, "h": 0.5,
        "title": "Translation readout",
        "text": "Assess sample-level\npresence of DC/APC\nprograms in plaque"
    },
]

for box in boxes:
    patch = FancyBboxPatch(
        box["xy"], box["w"], box["h"],
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0, edgecolor="0.5", facecolor="white",
        transform=ax1.transAxes
    )
    ax1.add_patch(patch)
    ax1.text(box["xy"][0] + 0.015, box["xy"][1] + 0.34, box["title"],
             transform=ax1.transAxes, fontsize=10, fontweight="bold", va="center")
    ax1.text(box["xy"][0] + 0.015, box["xy"][1] + 0.17, box["text"],
             transform=ax1.transAxes, fontsize=9, va="center")

arrow_y = 0.50
for x_start, x_end in [(0.24, 0.29), (0.51, 0.56), (0.74, 0.79)]:
    ax1.annotate(
        "", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
        xycoords=ax1.transAxes, textcoords=ax1.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.2)
    )

# Panel B: cDC2-like sample plot
ax2 = fig.add_subplot(gs[1, 0])

x = range(len(df))
y = pd.to_numeric(df["cdc2_score_mean"], errors="coerce")
status_colors = {"Asymptomatic": "#4C78A8", "Symptomatic": "#E45756"}
colors = [status_colors[s] for s in df["status"]]

ax2.bar(list(x), y, color=colors, edgecolor="black", linewidth=0.8)
ax2.set_xticks(list(x))
ax2.set_xticklabels(df["sample"], rotation=45, ha="right")
ax2.set_ylabel("Sample-level mean cDC2-like score")
ax2.set_title("B  Plaque cDC2-like antigen-presentation signal", loc="left", fontweight="bold")

for xi, yi, st in zip(range(len(df)), y, df["status"]):
    ax2.text(xi, yi + 0.02, "ASym" if st == "Asymptomatic" else "Sym",
             ha="center", va="bottom", fontsize=8)

ax2.text(0.02, 0.97, "Blue: asymptomatic\nRed: symptomatic",
         transform=ax2.transAxes, ha="left", va="top", fontsize=9)

# Panel C: summary table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis("off")
ax3.set_title("C  Sample-level DC/APC translation summary", loc="left", fontweight="bold", pad=8)

tbl = ax3.table(
    cellText=summary_df.values,
    colLabels=["Sample", "Status", "Cells", "cDC2-like", "pDC-like", "pDC−cDC2"],
    loc="center",
    cellLoc="center",
    colLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.08, 1.5)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_linewidth(0.8)
    else:
        cell.set_linewidth(0.5)

for r in range(1, len(summary_df) + 1):
    status = summary_df.iloc[r - 1]["status"]
    if status == "Asymptomatic":
        tbl[(r, 1)].set_facecolor("#D9E6F2")
    else:
        tbl[(r, 1)].set_facecolor("#F6D7D5")

fig.suptitle("Plaque-side translation of the dendritic-cell axis in GSE253902", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout()

png = figdir / "Figure_tier3_DC_presence_mapping_GSE253902.png"
pdf = figdir / "Figure_tier3_DC_presence_mapping_GSE253902.pdf"
csv = outdir / "GSE253902_dc_presence_sample_scores_for_figure.csv"

summary_df.to_csv(csv, index=False)
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(png)
print(pdf)
print(csv)
