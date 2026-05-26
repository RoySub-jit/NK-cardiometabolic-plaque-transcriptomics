from pathlib import Path
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures/final_main"
figdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

# -----------------------------
# Nodes
# -----------------------------
nodes = [
    (1.0, 2.0, "Higher\nnon-HDL"),
    (3.3, 2.0, "Circulating\nGZMK+/NKG7+\nNK state"),
    (5.8, 2.0, "Cytotoxic-\ninflammatory\nprogram"),
    (8.4, 2.0, "Symptomatic plaque /\nIPH enrichment"),
]

for x, y, text in nodes:
    box = plt.Rectangle(
        (x-0.9, y-0.55),
        1.8,
        1.1,
        fill=False,
        linewidth=2.0
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

# -----------------------------
# Arrows
# -----------------------------
arrowprops = dict(
    arrowstyle="->",
    linewidth=2.2,
    shrinkA=0,
    shrinkB=0,
)

ax.annotate("", xy=(2.4,2.0), xytext=(1.9,2.0), arrowprops=arrowprops)
ax.annotate("", xy=(4.9,2.0), xytext=(4.2,2.0), arrowprops=arrowprops)
ax.annotate("", xy=(7.5,2.0), xytext=(6.7,2.0), arrowprops=arrowprops)

# -----------------------------
# Supporting annotations
# -----------------------------
ax.text(
    3.3, 0.85,
    "GZMK, NKG7, GNLY,\nKLRD1, TYROBP",
    ha="center",
    fontsize=10
)

ax.text(
    5.8, 0.85,
    "PRF1, GZMB, CCL3/4,\nSTAT1, OAS1",
    ha="center",
    fontsize=10
)

ax.text(
    8.4, 0.85,
    "Cross-cohort convergence\nacross healthy, PBMC,\nand plaque datasets",
    ha="center",
    fontsize=10
)

ax.set_title(
    "Figure 6. Proposed model linking cardiometabolic variation to conserved NK-associated vascular inflammatory programs",
    fontsize=14,
    pad=20
)

fig.tight_layout()

fig.savefig(
    figdir / "Figure_6_conceptual_model.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    figdir / "Figure_6_conceptual_model.pdf",
    bbox_inches="tight"
)

print("Created Figure 6 conceptual model")
print(figdir)
