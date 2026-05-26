from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv").copy()
df = df[df["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()

df["lip.cholesterol_non_hdl"] = pd.to_numeric(df["lip.cholesterol_non_hdl"], errors="coerce")
df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
df = df.dropna(subset=["lip.cholesterol_non_hdl", "proportion"]).copy()

df["nonhdl_tertile"] = pd.qcut(df["lip.cholesterol_non_hdl"], q=3, labels=["Low", "Mid", "High"])
df = df.sort_values("proportion").reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

color_map = {"Low": "#dbe9f6", "Mid": "#8fbfe0", "High": "#2f6fb0"}
colors = [color_map[str(x)] for x in df["nonhdl_tertile"]]

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

fig, ax = plt.subplots(figsize=(7.4, 5.1))

ax.bar(df["rank"], df["proportion"], color=colors, edgecolor="black", linewidth=0.25)
ax.set_title("G", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Donors ranked by GZMK+ CD56dim NK proportion")
ax.set_ylabel("GZMK+ CD56dim NK proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [Patch(facecolor=color_map[k], edgecolor="black", label=k) for k in ["Low", "Mid", "High"]]
ax.legend(handles=legend_handles, frameon=False, title="Non-HDL tertile", loc="upper left")

png = figdir / "PanelG_NK_ranked_donors.png"
pdf = figdir / "PanelG_NK_ranked_donors.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
