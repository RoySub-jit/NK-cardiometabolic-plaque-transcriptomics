from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore

PROJECT = Path("/pub/subhajr/NK_cardio_GSE198339_reaudit")
OLD = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
OUT = PROJECT / "results/corrected_figure3_plaque_only_DEFG"
OUT.mkdir(parents=True, exist_ok=True)

H5AD = OLD / "results/tier2_validation/GSE224273_merged_rna_raw.h5ad"

if not H5AD.exists():
    raise FileNotFoundError(f"Missing plaque object: {H5AD}")

adata = sc.read_h5ad(H5AD).copy()

if "status" not in adata.obs.columns:
    raise RuntimeError("Required clinical-group column 'status' was not found in the plaque object.")

adata.obs["status"] = adata.obs["status"].astype(str)

sample_candidates = [
    "sample", "Sample", "sample_id", "SampleID",
    "orig.ident", "donor", "Donor", "patient", "Patient"
]
sample_col = next((c for c in sample_candidates if c in adata.obs.columns), None)

if sample_col is None:
    print("Available obs columns:", list(adata.obs.columns))
    raise RuntimeError("Sample column not automatically identified. Add the correct name to sample_candidates.")

print("Input plaque object:", H5AD)
print("Object shape:", adata.shape)
print("Sample column:", sample_col)
print("Clinical groups:")
print(adata.obs["status"].value_counts().to_string())

# -------------------------------------------------------------------------
# Normalization and NK-high plaque-cell definition
# -------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

nk_seed_genes = ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"]
nk_seed_genes = [g for g in nk_seed_genes if g in adata.var_names]

if len(nk_seed_genes) < 3:
    raise RuntimeError(f"Insufficient NK seed genes found: {nk_seed_genes}")

sc.tl.score_genes(adata, nk_seed_genes, score_name="nk_seed_score", use_raw=False)
nk_threshold = adata.obs["nk_seed_score"].quantile(0.90)
adata.obs["is_nk_high"] = adata.obs["nk_seed_score"] >= nk_threshold

nk = adata[adata.obs["is_nk_high"]].copy()
nk.obs["status"] = nk.obs["status"].astype(str)

print("NK-high threshold:", nk_threshold)
print("NK-high cells:", nk.n_obs)
print("NK-high cells by group:")
print(nk.obs["status"].value_counts().to_string())

# -------------------------------------------------------------------------
# Gene and module definitions: aligned with final supplementary workbook
# -------------------------------------------------------------------------
genes_for_panel_d = ["GZMK", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "FCGR3A", "TYROBP"]

modules = {
    "GZMK-like score": ["GZMK", "NKG7", "GNLY", "KLRD1"],
    "Cytotoxic core score": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"],
    "NK core score": ["NKG7", "GNLY", "KLRD1", "TYROBP"],
    "Cytotoxic/NK effector": ["NKG7", "GNLY", "PRF1", "GZMB", "KLRD1", "FCGR3A", "TYROBP"],
}

genes_for_panel_d = [g for g in genes_for_panel_d if g in nk.var_names]

for module, genes in list(modules.items()):
    present = [g for g in genes if g in nk.var_names]
    if len(present) != len(genes):
        raise RuntimeError(f"Missing genes for module {module}: expected {genes}, found {present}")
    modules[module] = present

def expr_vector(obj, gene):
    x = obj[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()

for gene in sorted(set(genes_for_panel_d + sum(modules.values(), []))):
    nk.obs[f"{gene}_expr"] = expr_vector(nk, gene)

for module, genes in modules.items():
    nk.obs[module] = nk.obs[[f"{g}_expr" for g in genes]].mean(axis=1)

# -------------------------------------------------------------------------
# Aggregate values at sample level
# -------------------------------------------------------------------------
required_columns = (
    [f"{g}_expr" for g in genes_for_panel_d]
    + list(modules.keys())
)

sample_status = (
    nk.obs.groupby(sample_col, observed=True)
    .first()[["status"]]
)

sample_values = (
    nk.obs.groupby(sample_col, observed=True)[required_columns]
    .mean()
    .join(sample_status)
    .reset_index()
)

sample_values = sample_values[
    sample_values["status"].isin(["Asymptomatic", "Symptomatic"])
].copy()

order_map = {"Asymptomatic": 0, "Symptomatic": 1}
sample_values["group_order"] = sample_values["status"].map(order_map)
sample_values = sample_values.sort_values(["group_order", sample_col]).reset_index(drop=True)

n_asym = int((sample_values["status"] == "Asymptomatic").sum())
n_sym = int((sample_values["status"] == "Symptomatic").sum())

if n_asym != 5 or n_sym != 2:
    print("WARNING: expected 5 asymptomatic and 2 symptomatic plaque samples.")
    print(sample_values[[sample_col, "status"]].to_string(index=False))

# -------------------------------------------------------------------------
# Exact permutation test at the sample level
# -------------------------------------------------------------------------
def exact_permutation_test(df, value_col):
    values = df[value_col].to_numpy(dtype=float)
    status = df["status"].to_numpy()

    asym_idx = np.where(status == "Asymptomatic")[0]
    sym_idx = np.where(status == "Symptomatic")[0]

    observed = values[asym_idx].mean() - values[sym_idx].mean()

    all_indices = np.arange(len(values))
    perm_effects = []

    for chosen_asym in combinations(all_indices, len(asym_idx)):
        chosen_asym = np.array(chosen_asym)
        chosen_sym = np.array([i for i in all_indices if i not in set(chosen_asym)])
        effect = values[chosen_asym].mean() - values[chosen_sym].mean()
        perm_effects.append(effect)

    perm_effects = np.asarray(perm_effects)
    p_value = np.mean(np.abs(perm_effects) >= abs(observed) - 1e-12)

    return observed, p_value, perm_effects

gene_results = []
for gene in genes_for_panel_d:
    effect, p_value, _ = exact_permutation_test(sample_values, f"{gene}_expr")
    gene_results.append({
        "Feature": gene,
        "Feature_type": "Gene expression",
        "Effect_Asym_minus_Sym": effect,
        "Exact_permutation_p": p_value,
        "Status": "Nominally supported" if p_value < 0.05 else (
            "Trend" if p_value < 0.10 else "Not significant"
        )
    })

module_results = []
for module in modules:
    effect, p_value, _ = exact_permutation_test(sample_values, module)
    module_results.append({
        "Feature": module,
        "Feature_type": "Module score",
        "Effect_Asym_minus_Sym": effect,
        "Exact_permutation_p": p_value,
        "Status": "Nominally supported" if p_value < 0.05 else (
            "Trend" if p_value < 0.10 else "Not significant"
        )
    })

gene_stats = pd.DataFrame(gene_results)
module_stats = pd.DataFrame(module_results)

all_stats = pd.concat([gene_stats, module_stats], ignore_index=True)
all_stats.to_csv(OUT / "Figure3_DEFG_plaque_only_exact_permutation_statistics.csv", index=False)
sample_values.to_csv(OUT / "Figure3_DEFG_plaque_only_sample_values.csv", index=False)

print("\nGene-level plaque statistics:")
print(gene_stats.to_string(index=False))
print("\nModule-level plaque statistics:")
print(module_stats.to_string(index=False))

# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

# -------------------------------------------------------------------------
# Panel D: gene-level plaque effects
# -------------------------------------------------------------------------
plot_d = gene_stats.sort_values("Effect_Asym_minus_Sym").copy()

fig, ax = plt.subplots(figsize=(5.4, 4.6))
y = np.arange(len(plot_d))

colors = [
    "#B37A27" if status == "Trend" else "#557A95"
    for status in plot_d["Status"]
]

ax.barh(y, plot_d["Effect_Asym_minus_Sym"], color=colors, alpha=0.9)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels(plot_d["Feature"])
ax.set_xlabel("Mean difference (asymptomatic - symptomatic)")
ax.set_title("D. Plaque NK-high gene-level effects", loc="left", fontweight="bold")

for yi, (_, row) in enumerate(plot_d.iterrows()):
    xpos = row["Effect_Asym_minus_Sym"]
    align = "left" if xpos >= 0 else "right"
    shift = 0.015 if xpos >= 0 else -0.015
    ax.text(
        xpos + shift,
        yi,
        f"p={row['Exact_permutation_p']:.4f}",
        ha=align,
        va="center",
        fontsize=8
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "Figure3D_plaque_gene_level_effects")

# -------------------------------------------------------------------------
# Panel E: plaque module-score effects
# -------------------------------------------------------------------------
plot_e = module_stats.sort_values("Effect_Asym_minus_Sym").copy()

fig, ax = plt.subplots(figsize=(5.8, 3.7))
y = np.arange(len(plot_e))

bar_colors = [
    "#B37A27" if status == "Trend" else "#557A95"
    for status in plot_e["Status"]
]

ax.barh(y, plot_e["Effect_Asym_minus_Sym"], color=bar_colors, alpha=0.9)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels(plot_e["Feature"])
ax.set_xlabel("Mean difference (asymptomatic - symptomatic)")
ax.set_title("E. Plaque NK/cytotoxic module effects", loc="left", fontweight="bold")

for yi, (_, row) in enumerate(plot_e.iterrows()):
    xpos = row["Effect_Asym_minus_Sym"]
    align = "left" if xpos >= 0 else "right"
    shift = 0.015 if xpos >= 0 else -0.015
    ax.text(
        xpos + shift,
        yi,
        f"p={row['Exact_permutation_p']:.4f}",
        ha=align,
        va="center",
        fontsize=8
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "Figure3E_plaque_module_effects")

# -------------------------------------------------------------------------
# Panel F: per-sample plaque module-score heatmap
# -------------------------------------------------------------------------
matrix = sample_values[list(modules.keys())].to_numpy(dtype=float)
matrix_z = zscore(matrix, axis=0, nan_policy="omit")
matrix_z = np.nan_to_num(matrix_z)

fig, ax = plt.subplots(figsize=(6.5, 4.1))
im = ax.imshow(matrix_z.T, aspect="auto", cmap="RdBu_r", vmin=-1.75, vmax=1.75)

sample_labels = [
    f"{row[sample_col]} | {row['status']}"
    for _, row in sample_values.iterrows()
]

ax.set_xticks(range(len(sample_labels)))
ax.set_xticklabels(sample_labels, rotation=45, ha="right")
ax.set_yticks(range(len(modules)))
ax.set_yticklabels(list(modules.keys()))
ax.set_title("F. Plaque sample-level module-score patterns", loc="left", fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Relative module score")

save(fig, "Figure3F_plaque_sample_module_heatmap")

# -------------------------------------------------------------------------
# Panel G: transparent plaque-only evidence status
# -------------------------------------------------------------------------
focus_rows = [
    ("GZMK-like score", module_stats.loc[module_stats["Feature"] == "GZMK-like score"].iloc[0]),
    ("Cytotoxic core score", module_stats.loc[module_stats["Feature"] == "Cytotoxic core score"].iloc[0]),
    ("NK core score", module_stats.loc[module_stats["Feature"] == "NK core score"].iloc[0]),
    ("Cytotoxic/NK effector", module_stats.loc[module_stats["Feature"] == "Cytotoxic/NK effector"].iloc[0]),
]

fig, ax = plt.subplots(figsize=(6.8, 3.5))
ax.axis("off")

ax.text(
    0.0, 0.98,
    "G. Plaque-only interpretation of NK/cytotoxic features",
    fontsize=11,
    fontweight="bold",
    va="top"
)

header_y = 0.79
row_h = 0.15
widths = [0.38, 0.20, 0.20, 0.22]
headers = ["Feature", "Effect", "p value", "Interpretation"]
x_positions = np.cumsum([0] + widths[:-1]).tolist()

for x, w, header in zip(x_positions, widths, headers):
    ax.add_patch(Rectangle((x, header_y), w, row_h, facecolor="#E6E6E6", edgecolor="white"))
    ax.text(x + w / 2, header_y + row_h / 2, header, ha="center", va="center",
            fontsize=8.5, fontweight="bold")

for i, (label, row) in enumerate(focus_rows):
    y0 = header_y - (i + 1) * row_h
    p = float(row["Exact_permutation_p"])
    effect = float(row["Effect_Asym_minus_Sym"])

    if p < 0.05:
        fill = "#D7ECEF"
        interpretation = "Supported"
    elif p < 0.10:
        fill = "#F5E5C8"
        interpretation = "Positive trend"
    else:
        fill = "#F0F0F0"
        interpretation = "Not significant"

    vals = [
        label,
        f"{effect:.3f}",
        f"{p:.4f}",
        interpretation
    ]

    for x, w, value in zip(x_positions, widths, vals):
        ax.add_patch(Rectangle((x, y0), w, row_h, facecolor=fill, edgecolor="white"))
        ax.text(x + w / 2, y0 + row_h / 2, value, ha="center", va="center", fontsize=8.2)

ax.text(
    0.0, 0.04,
    "Effect direction: asymptomatic minus symptomatic. "
    "All values derived from plaque single-cell data only.",
    fontsize=8.5
)

save(fig, "Figure3G_plaque_only_evidence_status")

# -------------------------------------------------------------------------
# Provenance note
# -------------------------------------------------------------------------
with open(OUT / "Figure3_DEFG_plaque_only_provenance.txt", "w") as handle:
    handle.write("Figure 3D-G plaque-only replacement panels\n")
    handle.write("=" * 52 + "\n")
    handle.write("No PBMC data, PBMC cluster comparisons, or PBMC pathway values are included.\n")
    handle.write(f"Input plaque object: {H5AD}\n")
    handle.write(f"Sample column: {sample_col}\n")
    handle.write(f"NK-high threshold: upper decile of score derived from {', '.join(nk_seed_genes)}\n")
    handle.write("Group comparison: asymptomatic minus symptomatic plaque samples.\n")
    handle.write("Statistics: two-sided exact label-permutation test at sample level.\n\n")
    handle.write("Module definitions:\n")
    for module, genes in modules.items():
        handle.write(f"- {module}: {', '.join(genes)}\n")
    handle.write("\nModule results:\n")
    handle.write(module_stats.to_string(index=False))
    handle.write("\n")

print("\nPASS: Plaque-only Figure 3D-G panels generated.")
for path in sorted(OUT.glob("*")):
    print(path.name)
