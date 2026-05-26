import scanpy as sc
from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
dataset_dir = project_dir / "data" / "first_dataset"
raw_dir = dataset_dir / "raw"
processed_dir = dataset_dir / "processed"
metadata_dir = dataset_dir / "metadata"
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"

processed_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

sc.settings.figdir = str(figures_dir)
sc.settings.verbosity = 3

# ----------------------------
# Input dataset
# ----------------------------
input_file = raw_dir / "input_dataset.h5ad"

if input_file.exists():
    print(f"Reading dataset from: {input_file}")
    adata = sc.read_h5ad(input_file)
else:
    print("No real dataset found yet in first_dataset/raw/. Using PBMC3k as temporary fallback.")
    adata = sc.datasets.pbmc3k()

# ----------------------------
# Basic preprocessing
# ----------------------------
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# ----------------------------
# Normalization and HVGs
# ----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable].copy()

# ----------------------------
# Dimensionality reduction and clustering
# ----------------------------
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, flavor="igraph", directed=False, n_iterations=2)

# ----------------------------
# Save outputs
# ----------------------------
output_h5ad = results_dir / "first_dataset_processed.h5ad"
adata.write(output_h5ad)

sc.pl.umap(adata, color=["leiden"], save="_first_dataset_leiden.png", show=False)

print("Done")
print(adata)
print(f"Saved object: {output_h5ad}")
print(f"Saved figure folder: {figures_dir}")
