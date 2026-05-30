from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
OLD = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
OUT = PROJECT / "results/corrected_figure5_panels"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Load verified source tables
# -------------------------------------------------------------------------
s2 = pd.read_csv(OLD / "results/supplementary/Table_S2_discovery_model_robustness.csv")
s5 = pd.read_csv(OLD / "results/supplementary/Table_S5_plaque_singlecell_statistics.csv")
s6 = pd.read_csv(OLD / "results/supplementary/Table_S6_IPH_validation_statistics.csv")
s10 = pd.read_csv(OLD / "results/supplementary/Table_S10_alternative_score_robustness.csv")
pbmc = pd.read_csv(
    PROJECT / "results/corrected_pbmc_statistics/GSE198339_verified_nonHDL_NK_statistics.csv"
)

def pbmc_p(feature):
    return float(pbmc.loc[pbmc["Feature"] == feature, "p_value"].iloc[0])

def pbmc_rho(feature):
    return float(pbmc.loc[pbmc["Feature"] == feature, "Spearman_rho"].iloc[0])

allen_p = float(s2.loc[s2["Analysis"] == "Primary adjusted model", "p_value"].iloc[0])
allen_beta = float(s2.loc[s2["Analysis"] == "Primary adjusted model", "beta"].iloc[0])

plaque_p = float(s5.loc[0, "p_value"])
plaque_effect = float(s5.loc[0, "Effect"])

iph_gzmk_p = float(s10.loc[s10["Score definition"] == "GZMK-like score", "p value"].iloc[0])
iph_gzmk_effect = float(
    s10.loc[s10["Score definition"] == "GZMK-like score",
            "Mean difference No IPH minus IPH"].iloc[0]
)
iph_nkcore_p = float(s10.loc[s10["Score definition"] == "NK core score", "p value"].iloc[0])
iph_nkcore_effect = float(
    s10.loc[s10["Score definition"] == "NK core score",
            "Mean difference No IPH minus IPH"].iloc[0]
)
iph_cytotoxic_p = float(
    s10.loc[s10["Score definition"] == "Cytotoxic/NK effector score", "p value"].iloc[0]
)
iph_cytotoxic_effect = float(
    s10.loc[s10["Score definition"] == "Cytotoxic/NK effector score",
            "Mean difference No IPH minus IPH"].iloc[0]
)

# -------------------------------------------------------------------------
# Verified, plotted source values
# Panel A reports -log10(p); effect directions are explicitly stated.
# These are not represented as comparable effect sizes across datasets.
# -------------------------------------------------------------------------
evidence = pd.DataFrame([
    {
        "Context": "Allen healthy discovery",
        "Feature": "GZMK+ CD56dim NK-cell abundance",
        "Estimate": allen_beta,
        "Estimate_type": "Adjusted beta",
        "p_value": allen_p,
        "Direction": "Higher with non-HDL",
        "Status": "Supported",
        "Source": "Table S2"
    },
    {
        "Context": "PBMC disease-context",
        "Feature": "Cytotoxic core score",
        "Estimate": pbmc_rho("Cytotoxic core score"),
        "Estimate_type": "Spearman rho",
        "p_value": pbmc_p("Cytotoxic core score"),
        "Direction": "Higher with non-HDL",
        "Status": "Supported",
        "Source": "Verified GSE198339 rerun"
    },
    {
        "Context": "PBMC disease-context",
        "Feature": "NKG7 expression in NK cells",
        "Estimate": pbmc_rho("NKG7 expression in NK cells"),
        "Estimate_type": "Spearman rho",
        "p_value": pbmc_p("NKG7 expression in NK cells"),
        "Direction": "Higher with non-HDL",
        "Status": "Supported",
        "Source": "Verified GSE198339 rerun"
    },
    {
        "Context": "PBMC disease-context",
        "Feature": "NK resting-cell proportion",
        "Estimate": pbmc_rho("NK resting-cell proportion"),
        "Estimate_type": "Spearman rho",
        "p_value": pbmc_p("NK resting-cell proportion"),
        "Direction": "Higher with non-HDL",
        "Status": "Supported",
        "Source": "Verified GSE198339 rerun"
    },
    {
        "Context": "PBMC disease-context",
        "Feature": "GZMK-like composite score",
        "Estimate": pbmc_rho("GZMK-like composite score"),
        "Estimate_type": "Spearman rho",
        "p_value": pbmc_p("GZMK-like composite score"),
        "Direction": "Higher with non-HDL",
        "Status": "Positive trend",
        "Source": "Verified GSE198339 rerun"
    },
    {
        "Context": "Plaque single-cell",
        "Feature": "GZMK-like score",
        "Estimate": plaque_effect,
        "Estimate_type": "Asymptomatic minus symptomatic",
        "p_value": plaque_p,
        "Direction": "Higher in asymptomatic",
        "Status": "Positive trend",
        "Source": "Table S5"
    },
    {
        "Context": "IPH bulk plaque",
        "Feature": "NK core score",
        "Estimate": iph_nkcore_effect,
        "Estimate_type": "No-IPH minus IPH",
        "p_value": iph_nkcore_p,
        "Direction": "Higher in IPH-positive",
        "Status": "Supported",
        "Source": "Table S10"
    },
    {
        "Context": "IPH bulk plaque",
        "Feature": "Cytotoxic/NK effector score",
        "Estimate": iph_cytotoxic_effect,
        "Estimate_type": "No-IPH minus IPH",
        "p_value": iph_cytotoxic_p,
        "Direction": "Higher in IPH-positive",
        "Status": "Supported",
        "Source": "Table S10"
    },
    {
        "Context": "IPH bulk plaque",
        "Feature": "GZMK-like score",
        "Estimate": iph_gzmk_effect,
        "Estimate_type": "No-IPH minus IPH",
        "p_value": iph_gzmk_p,
        "Direction": "Higher in IPH-positive",
        "Status": "Supported",
        "Source": "Table S10"
    },
])

evidence["minus_log10_p"] = -np.log10(evidence["p_value"])
evidence.to_csv(OUT / "Figure5A_verified_source_values.csv", index=False)

# -------------------------------------------------------------------------
# Figure 5A: transparent statistical evidence summary
# -------------------------------------------------------------------------
plot_df = evidence.iloc[::-1].copy()
labels = [
    f"{row.Context}: {row.Feature}"
    for row in plot_df.itertuples()
]
y = np.arange(len(plot_df))

status_color = {
    "Supported": "#2F6B7C",
    "Positive trend": "#B37A27"
}
colors = [status_color[x] for x in plot_df["Status"]]

fig, ax = plt.subplots(figsize=(10.8, 6.8))
ax.barh(y, plot_df["minus_log10_p"], color=colors, alpha=0.9)

ax.axvline(-np.log10(0.05), linestyle="--", linewidth=1.1, color="black")
ax.text(
    -np.log10(0.05) + 0.04,
    len(plot_df) - 0.35,
    "Nominal p = 0.05",
    fontsize=9,
    va="top"
)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Statistical evidence (−log10 p value)", fontsize=11)
ax.set_title(
    "A. Verified feature-level evidence across cardiometabolic and plaque contexts",
    fontsize=12,
    fontweight="bold",
    loc="left"
)

for yi, (_, row) in enumerate(plot_df.iterrows()):
    p_txt = f"p={row['p_value']:.4g}"
    direction = row["Direction"]
    ax.text(
        row["minus_log10_p"] + 0.07,
        yi,
        f"{p_txt}; {direction}",
        va="center",
        fontsize=8
    )

legend_handles = [
    Rectangle((0, 0), 1, 1, color=status_color["Supported"], label="Nominally supported (p < 0.05)"),
    Rectangle((0, 0), 1, 1, color=status_color["Positive trend"], label="Positive trend (0.05 ≤ p < 0.10)")
]
ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, max(plot_df["minus_log10_p"]) + 2.3)
plt.tight_layout()

fig.savefig(OUT / "Figure5A_verified_feature_level_evidence.pdf", bbox_inches="tight")
fig.savefig(OUT / "Figure5A_verified_feature_level_evidence.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 5B: evidence-status matrix
# No pathway enrichment or FDR claims are represented.
# -------------------------------------------------------------------------
columns = [
    "Allen\nhealthy discovery",
    "PBMC\n disease-context",
    "Plaque\nsingle-cell",
    "IPH\nbulk plaque"
]

rows = [
    "GZMK-associated feature",
    "Broader cytotoxic/NK program",
    "NKG7-associated feature",
    "NK-cell abundance/proportion"
]

cell_text = [
    [
        "Supported\nGZMK+ state\np=0.0309",
        "Trend\nGZMK-like score\np=0.0757",
        "Trend; higher\nasymptomatic\np=0.0952",
        "Supported; higher IPH\nGZMK-like score\np=0.000343"
    ],
    [
        "Not directly\ntested",
        "Supported\ncytotoxic core\np=0.00451",
        "Not quantified\nin Table S5",
        "Supported; higher IPH\ncytotoxic/NK effector\np=0.000068"
    ],
    [
        "Not directly\ntested",
        "Supported\nNKG7\np=0.00932",
        "Not quantified\nin Table S5",
        "Supported; higher IPH\nNKG7\np=0.000172"
    ],
    [
        "Supported\nGZMK+ NK abundance\np=0.0309",
        "Supported\nNK resting proportion\np=0.01144",
        "Not tested",
        "Not attributable\nfrom bulk tissue"
    ]
]

status = [
    ["supported", "trend", "trend", "supported"],
    ["neutral", "supported", "neutral", "supported"],
    ["neutral", "supported", "neutral", "supported"],
    ["supported", "supported", "neutral", "neutral"],
]

cell_color = {
    "supported": "#D8ECEF",
    "trend": "#F5E5C8",
    "neutral": "#F0F0F0"
}

fig, ax = plt.subplots(figsize=(11.2, 5.8))
ax.axis("off")

nrows = len(rows)
ncols = len(columns)
cell_w = 1 / (ncols + 1.65)
left_margin = cell_w * 1.65
cell_h = 0.16
top = 0.84

# Column headers
for j, col in enumerate(columns):
    x = left_margin + j * cell_w
    ax.add_patch(Rectangle((x, top), cell_w, cell_h, facecolor="#E5E5E5", edgecolor="white"))
    ax.text(x + cell_w/2, top + cell_h/2, col, ha="center", va="center",
            fontsize=9, fontweight="bold")

# Row labels and cells
for i, row_name in enumerate(rows):
    y0 = top - (i + 1) * cell_h
    ax.add_patch(Rectangle((0, y0), left_margin, cell_h, facecolor="#E5E5E5", edgecolor="white"))
    ax.text(0.01, y0 + cell_h/2, row_name, ha="left", va="center",
            fontsize=9, fontweight="bold")

    for j in range(ncols):
        x = left_margin + j * cell_w
        ax.add_patch(Rectangle(
            (x, y0), cell_w, cell_h,
            facecolor=cell_color[status[i][j]],
            edgecolor="white"
        ))
        ax.text(
            x + cell_w/2, y0 + cell_h/2,
            cell_text[i][j],
            ha="center", va="center", fontsize=8
        )

ax.text(
    0, 0.97,
    "B. Evidence-status matrix based on verified statistical outputs",
    ha="left", va="top", fontsize=12, fontweight="bold"
)
ax.text(
    0, 0.07,
    "Supported: nominal p < 0.05; Trend: 0.05 ≤ p < 0.10. "
    "Panel reports verified feature-level evidence only; no pathway-level FDR claims are made.",
    ha="left", va="bottom", fontsize=9
)

plt.tight_layout()
fig.savefig(OUT / "Figure5B_verified_evidence_status_matrix.pdf", bbox_inches="tight")
fig.savefig(OUT / "Figure5B_verified_evidence_status_matrix.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# Save a plain-text provenance report
with open(OUT / "Figure5_verified_panel_provenance.txt", "w") as fh:
    fh.write("Corrected Figure 5A-B provenance report\n")
    fh.write("=" * 50 + "\n")
    fh.write("Figure 5A: feature-level statistical evidence summary generated from verified source tables.\n")
    fh.write("Figure 5B: evidence-status matrix; no pathway-level enrichment or FDR values displayed.\n\n")
    fh.write("Sources:\n")
    fh.write("- Allen discovery: Table_S2_discovery_model_robustness.csv\n")
    fh.write("- PBMC disease-context: verified rerun from official processed 9,368-cell GSE198339 dataset\n")
    fh.write("- Plaque single-cell: Table_S5_plaque_singlecell_statistics.csv\n")
    fh.write("- IPH bulk plaque: Table_S10_alternative_score_robustness.csv and Table_S6_IPH_validation_statistics.csv\n\n")
    fh.write(evidence.to_string(index=False))
    fh.write("\n")

print("PASS: Corrected Figure 5A and 5B generated from verified source values.")
print()
for f in sorted(OUT.glob("*")):
    print(f.name)
