from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")

outdir = base / "results/supplementary"
figdir = base / "figures/supplementary"

outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ----------------------------
# Table S1
# ----------------------------

table_s1 = pd.DataFrame([
    ["Allen Human Immune Health Atlas",
     "Blood scRNA-seq",
     "Healthy adults",
     "GZMK+ CD56dim NK vs non-HDL",
     "n=74 donors",
     "Discovery"],

    ["GSE198339",
     "PBMC scRNA-seq",
     "External blood validation",
     "NK readouts vs non-HDL",
     "n=8; 13,787 cells",
     "PBMC validation"],

    ["GSE224273",
     "Plaque scRNA-seq",
     "Carotid plaque",
     "Asymptomatic vs symptomatic NK-high cells",
     "Asym n=5; Sym n=2",
     "Plaque single-cell translation"],

    ["GSE163154",
     "Bulk plaque transcriptomics",
     "Carotid plaque IPH",
     "No IPH vs IPH",
     "No IPH n=16; IPH n=27",
     "Independent plaque validation"]
],
columns=[
    "Dataset",
    "Data_type",
    "Context",
    "Primary_readout",
    "Sample_size",
    "Role"
])

table_s1.to_csv(
    outdir / "Table_S1_dataset_summary.csv",
    index=False
)

# ----------------------------
# Table S2
# ----------------------------

table_s2 = pd.DataFrame([
    ["Primary adjusted model",
     74,
     0.000364,
     0.0000345,
     0.0006930,
     0.0309,
     0.0897,
     0.0228],

    ["Leave-one-out median",
     74,
     0.000369,
     np.nan,
     np.nan,
     0.0286,
     np.nan,
     np.nan],

    ["Bootstrap median",
     1000,
     0.000371,
     np.nan,
     np.nan,
     np.nan,
     np.nan,
     np.nan]
],
columns=[
    "Analysis",
    "n",
    "beta",
    "CI_lower",
    "CI_upper",
    "p_value",
    "R2",
    "adjusted_R2"
])

table_s2.to_csv(
    outdir / "Table_S2_discovery_model_robustness.csv",
    index=False
)

# ----------------------------
# Table S3
# ----------------------------

table_s3 = pd.DataFrame([
    ["NKG7 expression", 0.850, 0.0075],
    ["NK resting proportion", 0.826, 0.0114],
    ["GZMK-like score", 0.814, 0.0138]
],
columns=["Readout", "rho", "p_value"])

table_s3.to_csv(
    outdir / "Table_S3_PBMC_validation.csv",
    index=False
)

# ----------------------------
# Figure S1
# ----------------------------

fig, ax = plt.subplots(figsize=(5,3.5))

ax.barh(table_s3["Readout"], table_s3["rho"])

for i, row in table_s3.iterrows():
    ax.text(
        row["rho"] + 0.02,
        i,
        f"ρ={row['rho']:.3f}, p={row['p_value']:.4f}",
        va="center",
        fontsize=8
    )

ax.set_xlabel("Spearman ρ")
ax.set_title("Fig. S1. PBMC validation summary")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    figdir / "Fig_S1_PBMC_validation_summary.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    figdir / "Fig_S1_PBMC_validation_summary.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ----------------------------
# Excel workbook
# ----------------------------

xlsx_path = outdir / "NK_cardio_plaque_supplementary_tables.xlsx"

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    table_s1.to_excel(writer, sheet_name="Table_S1", index=False)
    table_s2.to_excel(writer, sheet_name="Table_S2", index=False)
    table_s3.to_excel(writer, sheet_name="Table_S3", index=False)

print("Supplementary package created successfully")
print("Tables:", outdir)
print("Figures:", figdir)
print("Workbook:", xlsx_path)


# ============================================================
# Additional supplementary tables and figures
# ============================================================

# ---------- Table S4 ----------
table_s4 = pd.DataFrame([
    ["GSE198339", "Cluster 1 vs cluster 4", "GZMK", "Cluster 1", 1.5e-52],
    ["GSE198339", "Cluster 1 vs cluster 4", "NKG7", "Cluster 4", 9.8e-116],
    ["GSE198339", "Cluster 1 vs cluster 4", "GNLY", "Cluster 4", 3.4e-271],
    ["GSE198339", "Cluster 1 vs cluster 4", "FCGR3A", "Cluster 4", 3.6e-159],
    ["GSE198339", "Cluster 1 vs cluster 4", "TYROBP", "Cluster 4", 0.0],
],
columns=[
    "Dataset",
    "Comparison",
    "Gene",
    "Enriched_in",
    "p_value"
])

table_s4.to_csv(
    outdir / "Table_S4_cluster_marker_statistics.csv",
    index=False
)

# ---------- Table S5 ----------
table_s5 = pd.DataFrame([
    ["GSE224273",
     "GZMK-like score",
     "Asymptomatic vs symptomatic",
     5,
     2,
     0.241,
     0.0952]
],
columns=[
    "Dataset",
    "Feature",
    "Comparison",
    "n_asymptomatic",
    "n_symptomatic",
    "Effect",
    "p_value"
])

table_s5.to_csv(
    outdir / "Table_S5_plaque_singlecell_statistics.csv",
    index=False
)

# ---------- Table S6 ----------
table_s6 = pd.DataFrame([
    ["gzmk_like_score", -0.415, 3.43e-4],
    ["NKG7", -0.658, 1.72e-4],
    ["TYROBP", -1.514, 2.93e-7],
    ["GNLY", -0.234, 0.00227],
    ["KLRD1", -0.091, 0.00803],
    ["GZMK", np.nan, 0.0298]
],
columns=[
    "feature",
    "No_IPH_minus_IPH",
    "mannwhitney_p"
])

table_s6.to_csv(
    outdir / "Table_S6_IPH_validation_statistics.csv",
    index=False
)

# ---------- Table S7 ----------
modules = {
    "GZMK-like score": ["GZMK", "NKG7", "GNLY", "KLRD1"],
    "NK core score": ["NKG7", "GNLY", "KLRD1", "TYROBP"],
    "Cytotoxic/NK effector": ["NKG7", "GNLY", "PRF1", "GZMB", "KLRD1", "FCGR3A", "TYROBP"],
    "Inflammatory/cytokine": ["CCL3", "CCL4", "CCL5", "IL32", "TNF", "IFNG"],
    "Interferon response": ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1"],
}

table_s7 = pd.DataFrame([
    {
        "Module": k,
        "Genes": ", ".join(v),
        "n_genes": len(v)
    }
    for k, v in modules.items()
])

table_s7.to_csv(
    outdir / "Table_S7_curated_modules.csv",
    index=False
)

# ---------- Figure S2 ----------
fig, ax = plt.subplots(figsize=(5.4,3.8))

ax.barh(
    table_s6["feature"],
    table_s6["No_IPH_minus_IPH"]
)

for i, row in table_s6.iterrows():
    val = row["No_IPH_minus_IPH"]

    if pd.notna(val):
        ax.text(
            val,
            i,
            f" p={row['mannwhitney_p']:.2g}",
            va="center",
            fontsize=8,
            ha="right" if val < 0 else "left"
        )

ax.axvline(0, color="black", linewidth=1)

ax.set_xlabel("Mean difference (No IPH - IPH)")
ax.set_title("Fig. S2. IPH NK validation")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    figdir / "Fig_S2_IPH_validation.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    figdir / "Fig_S2_IPH_validation.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ---------- Figure S3 ----------
fig, ax = plt.subplots(figsize=(6.0,4.0))

ax.barh(
    table_s7["Module"],
    table_s7["n_genes"]
)

for i, row in table_s7.iterrows():
    ax.text(
        row["n_genes"] + 0.2,
        i,
        str(row["n_genes"]),
        va="center",
        fontsize=8
    )

ax.set_xlabel("Number of genes")
ax.set_title("Fig. S3. Curated module sizes")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    figdir / "Fig_S3_module_sizes.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    figdir / "Fig_S3_module_sizes.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ---------- Update Excel workbook ----------
with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table_s4.to_excel(writer, sheet_name="Table_S4", index=False)
    table_s5.to_excel(writer, sheet_name="Table_S5", index=False)
    table_s6.to_excel(writer, sheet_name="Table_S6", index=False)
    table_s7.to_excel(writer, sheet_name="Table_S7", index=False)

print("Extended supplementary package completed")

