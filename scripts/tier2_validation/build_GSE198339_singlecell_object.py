from pathlib import Path
import re
import scanpy as sc
import pandas as pd
import numpy as np

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
datadir = base / "data/tier2_validation/GSE198339/extracted"
outdir = base / "results/tier2_validation"
outdir.mkdir(parents=True, exist_ok=True)

h5_files = sorted(datadir.glob("GSM*_Participant_*_raw_gene_bc_matrices_h5.h5"))
if not h5_files:
    raise FileNotFoundError(f"No raw h5 files found in {datadir}")

meta_all = None
meta_file = datadir / "GSE198339_all_metadata_combined.csv"
if meta_file.exists():
    meta_all = pd.read_csv(meta_file)

adatas = []

for f in h5_files:
    m = re.search(r"(GSM\d+)_Participant_(\d+)_raw_gene_bc_matrices_h5\.h5$", f.name)
    if not m:
        raise ValueError(f"Could not parse file name: {f.name}")
    gsm = m.group(1)
    participant_num = int(m.group(2))
    participant_id = f"Participant_{participant_num}"

    ad = sc.read_10x_h5(f)
    ad.var_names_make_unique()

    # keep source info
    ad.obs["gsm"] = gsm
    ad.obs["participant_id"] = participant_id
    ad.obs["source_file"] = f.name

    # try to add participant-level metadata if available
    if meta_all is not None:
        tmp = meta_all.copy()
        # try common matching columns
        match_cols = [c for c in tmp.columns if c.lower() in ["gsm", "geo_accession", "participant", "participant_id", "sample", "sample_id"]]
        matched = None
        for c in match_cols:
            vals = tmp[c].astype(str)
            if gsm in vals.values:
                matched = tmp.loc[vals == gsm].head(1)
                break
            if participant_id in vals.values:
                matched = tmp.loc[vals == participant_id].head(1)
                break
        if matched is not None and len(matched) == 1:
            row = matched.iloc[0]
            for col in matched.columns:
                ad.obs[col] = str(row[col])

    adatas.append(ad)

print(f"Loaded {len(adatas)} participant objects")

# merge
adata = adatas[0].concatenate(
    *adatas[1:],
    batch_key="batch",
    batch_categories=[a.obs['participant_id'].iloc[0] for a in adatas],
    index_unique="-"
)

# clean up var names after concatenate
if "gene_ids" in adata.var.columns:
    pass

# basic QC metrics
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# light filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# keep a copy of raw counts
adata.layers["counts"] = adata.X.copy()

# normalize / log
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVG / PCA / neighbors / UMAP
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
adata = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)

# simple clustering
sc.tl.leiden(adata, resolution=0.6, key_added="leiden_r06")

# simple NK-related scores
nk_seed = [g for g in ["NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"] if g in adata.var_names]
gzmk_like = [g for g in ["GZMK", "NKG7", "GNLY", "KLRD1"] if g in adata.var_names]

if nk_seed:
    sc.tl.score_genes(adata, nk_seed, score_name="nk_seed_score", use_raw=False)
if gzmk_like:
    sc.tl.score_genes(adata, gzmk_like, score_name="gzmk_like_score", use_raw=False)

# save
out_h5ad = outdir / "GSE198339_merged_singlecell_umap.h5ad"
adata.write(out_h5ad)

# cluster summaries
obs = adata.obs.copy()
summary_cols = [c for c in ["participant_id", "gsm", "batch", "leiden_r06", "nk_seed_score", "gzmk_like_score"] if c in obs.columns]
obs[summary_cols].to_csv(outdir / "GSE198339_merged_singlecell_obs_summary.csv", index=False)

print("Saved:", out_h5ad)
print("Cells:", adata.n_obs, "Genes:", adata.n_vars)
print("Leiden clusters:")
print(adata.obs["leiden_r06"].value_counts().sort_index())
