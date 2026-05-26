from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset" / "panel_exports"
figdir.mkdir(parents=True, exist_ok=True)

# try detailed file first
cand = [
    base / "results/first_dataset/nk_primary_nonhdl_leave_one_out_details.csv",
    base / "results/first_dataset/nk_primary_nonhdl_leaveoneout_details.csv",
    base / "results/first_dataset/leave_one_out_details.csv",
]
detail_file = None
for c in cand:
    if c.exists():
        detail_file = c
        break

if detail_file is None:
    raise FileNotFoundError("Could not find leave-one-out detailed file.")

df = pd.read_csv(detail_file).copy()
beta_col = [c for c in df.columns if "beta" in c.lower()][0]
p_col = [c for c in df.columns if c.lower() in ["p_value", "p", "pval", "p_value_nonhdl"]]
p_col = p_col[0] if p_col else None

df[beta_col] = pd.to_numeric(df[beta_col], errors="coerce")
df = df.dropna(subset=[beta_col]).copy()
df = df.sort_values(beta_col).reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

beta_median = df[beta_col].median()

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

fig, ax = plt.subplots(figsize=(7.0, 5.3))

ax.scatter(df["rank"], df[beta_col], s=28, color="#2f78b7", edgecolors="black", linewidths=0.4)
ax.axhline(beta_median, color="black", linewidth=1.4)
ax.axhline(0, color="0.35", linestyle=":", linewidth=1.0)

ax.set_title("E", loc="left", fontweight="bold", fontsize=18)
ax.set_xlabel("Leave-one-out iteration")
ax.set_ylabel("Effect size")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

txt = f"n = {len(df)}\nMedian β = {beta_median:.6f}\nMin = {df[beta_col].min():.6f}\nMax = {df[beta_col].max():.6f}"
if p_col is not None:
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    txt += f"\nMedian P = {df[p_col].median():.4f}"

ax.text(
    0.03, 0.97, txt,
    transform=ax.transAxes, ha="left", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92)
)

png = figdir / "PanelE_NK_leaveoneout_full.png"
pdf = figdir / "PanelE_NK_leaveoneout_full.pdf"
fig.savefig(png, dpi=400, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print("Used file:", detail_file)
