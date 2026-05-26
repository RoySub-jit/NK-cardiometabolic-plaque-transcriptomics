from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import linregress

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
figdir = base / "figures" / "first_dataset"
figdir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
adata = sc.read_h5ad(base / "data/first_dataset/raw/human_immune_health_atlas_nk-ilc.h5ad")
nk = pd.read_csv(base / "results/first_dataset/nk_L3_with_clinical_by_sample.csv")
rob = pd.read_csv(base / "results/first_dataset/primary_robustness_checks_combined_summary.csv")
boot_details = pd.read_csv(base / "results/first_dataset/nk_primary_nonhdl_bootstrap_details.csv")

# ---------------------------
# Prep single-cell object
# ---------------------------
states_order = [
    "GZMK- CD56dim NK cell",
    "GZMK+ CD56dim NK cell",
    "Adaptive NK cell",
    "ISG+ CD56dim NK cell",
    "CD56bright NK cell",
    "Proliferating NK cell",
    "ILC",
]

adata = adata[adata.obs["AIFI_L3"].isin(states_order)].copy()
adata.obs["AIFI_L3"] = pd.Categorical(adata.obs["AIFI_L3"], categories=states_order, ordered=True)

# Normalize/log for marker visualization if needed
if adata.X.max() > 50:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ---------------------------
# Prep sample-level association
# ---------------------------
sub = nk[nk["AIFI_L3"] == "GZMK+ CD56dim NK cell"].copy()
sub["lip.cholesterol_non_hdl"] = pd.to_numeric(sub["lip.cholesterol_non_hdl"], errors="coerce")
sub["proportion"] = pd.to_numeric(sub["proportion"], errors="coerce")
sub = sub.dropna(subset=["lip.cholesterol_non_hdl", "proportion"]).copy()
sub["nonhdl_tertile"] = pd.qcut(sub["lip.cholesterol_non_hdl"], q=3, labels=["Low", "Mid", "High"])

lr = linregress(sub["lip.cholesterol_non_hdl"], sub["proportion"])
xline = np.linspace(sub["lip.cholesterol_non_hdl"].min(), sub["lip.cholesterol_non_hdl"].max(), 200)
yline = lr.intercept + lr.slope * xline

# ---------------------------
# Robustness rows
# ---------------------------
age_row = rob[rob["analysis"] == "NK_age_continuous"].iloc[0]
loo_row = rob[rob["analysis"] == "NK_leave_one_out"].iloc[0]
boot_row = rob[rob["analysis"] == "NK_bootstrap"].iloc[0]

# ---------------------------
# Style
# ---------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

state_colors = {
    "GZMK- CD56dim NK cell": "#9ecae1",
    "GZMK+ CD56dim NK cell": "#1f78b4",
    "Adaptive NK cell": "#fdb462",
    "ISG+ CD56dim NK cell": "#b2df8a",
    "CD56bright NK cell": "#cab2d6",
    "Proliferating NK cell": "#fb9a99",
    "ILC": "#bdbdbd",
}

genes = ["GZMK", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "FCGR3A", "TYROBP"]
genes = [g for g in genes if g in adata.var_names]

# ---------------------------
# Layout
# ---------------------------
fig = plt.figure(figsize=(14.2, 10.2))
gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.15, 1.05], hspace=0.62, wspace=0.55)

# ---------------------------
# Panel A: visual schematic
# ---------------------------
axA = fig.add_subplot(gs[0, :])
axA.axis("off")
axA.set_title("A", loc="left", fontweight="bold", pad=2)

boxes = [
    ((0.02, 0.20), 0.23, 0.56, "Healthy adult blood", "Allen atlas\n74 samples\nscRNA-seq + clinical labs"),
    ((0.29, 0.20), 0.24, 0.56, "Single-cell NK state", "GZMK+ CD56dim NK\nidentified at AIFI L3"),
    ((0.58, 0.20), 0.18, 0.56, "Primary association", "Higher non-HDL\ntracks higher\nGZMK+ CD56dim NK"),
    ((0.81, 0.20), 0.16, 0.56, "Robustness", "Adjusted model\nleave-one-out\nbootstrap"),
]
fills = ["#eef5fb", "#e8f2fb", "#fff4e6", "#f5f5f5"]

for (xy, w, h, title, txt), fc in zip(boxes, fills):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        linewidth=1.0, edgecolor="0.45", facecolor=fc,
        transform=axA.transAxes
    )
    axA.add_patch(patch)
    axA.text(xy[0] + 0.015, xy[1] + 0.37, title, transform=axA.transAxes,
             fontsize=10, fontweight="bold", va="center")
    axA.text(xy[0] + 0.015, xy[1] + 0.17, txt, transform=axA.transAxes,
             fontsize=9, va="center")

for x1, x2 in [(0.25, 0.29), (0.53, 0.58), (0.76, 0.81)]:
    axA.annotate("", xy=(x2, 0.48), xytext=(x1, 0.48),
                 xycoords=axA.transAxes, textcoords=axA.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.3, color="0.35"))

# ---------------------------
# Panel B: UMAP by AIFI_L3
# ---------------------------
axB = fig.add_subplot(gs[1, 0:2])
axB.set_title("B", loc="left", fontweight="bold")
umap = adata.obsm["X_umap"]

for state in states_order:
    idx = np.where(adata.obs["AIFI_L3"].astype(str).values == state)[0]
    if len(idx) == 0:
        continue
    axB.scatter(
        umap[idx, 0], umap[idx, 1],
        s=3.0,
        color=state_colors.get(state, "gray"),
        alpha=0.55 if state != "GZMK+ CD56dim NK cell" else 0.75,
        linewidths=0,
        label=state
    )

axB.set_xlabel("UMAP1")
axB.set_ylabel("UMAP2")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
leg = axB.legend(frameon=False, loc="best", fontsize=7.5, markerscale=2.0, handletextpad=0.4)
for lh in leg.legend_handles:
    try:
        lh.set_alpha(1)
    except Exception:
        pass

# ---------------------------
# Panel C: marker dot plot
# ---------------------------
axC = fig.add_subplot(gs[1, 2:4])
axC.set_title("C", loc="left", fontweight="bold")

# compute average expression and percent expressing
expr = adata[:, genes].to_df()
expr["AIFI_L3"] = adata.obs["AIFI_L3"].astype(str).values

avg_rows = []
pct_rows = []
for state in states_order:
    subdf = expr[expr["AIFI_L3"] == state]
    if len(subdf) == 0:
        continue
    avg_rows.append(subdf[genes].mean(axis=0).rename(state))
    pct_rows.append((subdf[genes] > 0).mean(axis=0).rename(state))

avg_df = pd.DataFrame(avg_rows).reindex(states_order)
pct_df = pd.DataFrame(pct_rows).reindex(states_order)

# z-scale by gene for visual comparability
avg_z = (avg_df - avg_df.mean(axis=0)) / avg_df.std(axis=0, ddof=0)
avg_z = avg_z.replace([np.inf, -np.inf], np.nan).fillna(0)

xpos = np.arange(len(genes))
ypos = np.arange(len(states_order))

for yi, state in enumerate(states_order):
    for xi, gene in enumerate(genes):
        axC.scatter(
            xi, yi,
            s=20 + 180 * pct_df.loc[state, gene],
            c=avg_z.loc[state, gene],
            cmap="Blues",
            vmin=-1.5, vmax=1.5,
            edgecolors="black",
            linewidths=0.25
        )

axC.set_xticks(xpos)
axC.set_xticklabels(genes, rotation=35, ha="right")
axC.set_yticks(ypos)
axC.set_yticklabels(states_order)
axC.invert_yaxis()
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)

# ---------------------------
# Panel D: scatter association
# ---------------------------
axD = fig.add_subplot(gs[2, 0])
axD.set_title("D", loc="left", fontweight="bold")
axD.scatter(sub["lip.cholesterol_non_hdl"], sub["proportion"], s=34, edgecolors="black", linewidths=0.5)
axD.plot(xline, yline, linewidth=1.3)
axD.set_xlabel("Non-HDL cholesterol")
axD.set_ylabel("GZMK+ CD56dim NK\nproportion")
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.text(
    0.03, 0.97,
    f"n = {len(sub)}\nP = {lr.pvalue:.4f}\nR² = {lr.rvalue**2:.3f}",
    transform=axD.transAxes, ha="left", va="top", fontsize=8.4,
    bbox=dict(boxstyle="round,pad=0.23", facecolor="white", edgecolor="0.8", alpha=0.92)
)

# ---------------------------
# Panel E: adjusted model + leave-one-out
# ---------------------------
axE = fig.add_subplot(gs[2, 1:3])
axE.set_title("E", loc="left", fontweight="bold")

# primary model row
axE.errorbar(
    [age_row["beta"]], [1],
    xerr=[[age_row["beta"] - age_row["conf_low"]], [age_row["conf_high"] - age_row["beta"]]],
    fmt='o', color='black', capsize=4
)

# leave-one-out interval row
axE.plot([loo_row["beta_min"], loo_row["beta_max"]], [0, 0], color="black", linewidth=1.5)
axE.scatter([loo_row["beta_median"]], [0], color="black", s=55, zorder=3)

axE.axvline(0, color="0.35", linestyle=":", linewidth=1.0)
axE.set_yticks([1, 0])
axE.set_yticklabels(["Adjusted model", "Leave-one-out"])
axE.set_xlabel("Effect size")
axE.spines["top"].set_visible(False)
axE.spines["right"].set_visible(False)
axE.text(
    0.03, 0.97,
    f"Adjusted β = {age_row['beta']:.6f}, P = {age_row['p_value']:.4f}\n"
    f"Leave-one-out median β = {loo_row['beta_median']:.6f}, median P = {loo_row['p_median']:.4f}",
    transform=axE.transAxes, ha="left", va="top", fontsize=8.4
)

# ---------------------------
# Panel F: bootstrap distribution
# ---------------------------
axF = fig.add_subplot(gs[2, 3])
axF.set_title("F", loc="left", fontweight="bold")
vals = pd.to_numeric(boot_details["beta"], errors="coerce").dropna().values

axF.hist(vals, bins=28, edgecolor="white")
axF.axvline(boot_row["beta_median"], color="black", linewidth=1.5)
axF.axvline(boot_row["beta_ci_low_2_5"], color="black", linestyle="--", linewidth=1.0)
axF.axvline(boot_row["beta_ci_high_97_5"], color="black", linestyle="--", linewidth=1.0)
axF.axvline(0, color="0.35", linestyle=":", linewidth=1.0)
axF.set_xlabel("Bootstrapped effect size")
axF.set_ylabel("Count")
axF.spines["top"].set_visible(False)
axF.spines["right"].set_visible(False)
axF.text(
    0.03, 0.97,
    f"n = {int(boot_row['n_bootstrap'])}\n95% CI [{boot_row['beta_ci_low_2_5']:.6f}, {boot_row['beta_ci_high_97_5']:.6f}]\n"
    f"Expected sign = {boot_row['prop_expected_sign']:.3f}",
    transform=axF.transAxes, ha="left", va="top", fontsize=8.1,
    bbox=dict(boxstyle="round,pad=0.23", facecolor="white", edgecolor="0.8", alpha=0.92)
)

fig.suptitle(
    "Figure 1. Discovery and robustness of a lipid-associated GZMK-like NK-cell axis in healthy adult blood",
    fontsize=15, fontweight="bold", y=0.99
)

# safer than tight_layout here
fig.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.98, hspace=0.72, wspace=0.55)

png = figdir / "Figure1_NK_discovery_robustness_v2.png"
pdf = figdir / "Figure1_NK_discovery_robustness_v2.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
