from pathlib import Path
import pandas as pd
import scanpy as sc
from scipy import io
from scipy.sparse import csr_matrix, vstack

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE253902/extracted")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

samples = {
    "RPE008": {"sample": "carotid_5", "status": "Symptomatic", "gsm": "GSM8029950"},
    "RPE010": {"sample": "carotid_7", "status": "Asymptomatic", "gsm": "GSM8029952"},
    "RPE012": {"sample": "carotid_8", "status": "Symptomatic", "gsm": "GSM8029954"},
    "RPE014": {"sample": "carotid_10", "status": "Symptomatic", "gsm": "GSM8029956"},
    "RPE015": {"sample": "carotid_11", "status": "Symptomatic", "gsm": "GSM8029958"},
    "RPE016": {"sample": "carotid_12", "status": "Symptomatic", "gsm": "GSM8029960"},
}

adatas = []

for code, meta in samples.items():
    gsm = meta["gsm"]
    feat = base / f"{gsm}_{code}_features.tsv.dup_marked.tsv.gz"
    bc = base / f"{gsm}_{code}_barcodes.tsv.gz"
    mtx = base / f"{gsm}_{code}_matrix.mtx.gz"

    features = pd.read_csv(feat, sep="\t", header=None, names=["ensembl", "gene", "feature_type"])
    barcodes = pd.read_csv(bc, sep="\t", header=None, names=["barcode"])
    X = io.mmread(mtx).tocsr().T  # cells x genes

    var = features.copy()
    var.index = var["gene"].astype(str)
    obs = pd.DataFrame(index=barcodes["barcode"].astype(str))
    obs["sample"] = meta["sample"]
    obs["status"] = meta["status"]
    obs["gsm"] = gsm

    ad = sc.AnnData(X=X, obs=obs, var=var)
    ad.var_names_make_unique()
    adatas.append(ad)

adata = sc.concat(adatas, join="outer", label="batch", keys=[x["sample"] for x in samples.values()], index_unique="-")
adata.write(outdir / "GSE253902_merged_gex_raw.h5ad")

# normalize and score modules
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cdc2_genes = ["HLA-DRA", "HLA-DRB1", "CD74", "FCER1A", "CD1C", "CLEC10A"]
pdc_genes = ["GZMB", "TCF4", "JCHAIN", "IRF7"]
myeloid_genes = ["LST1", "CTSS", "FCN1", "TYMP"]

present_cdc2 = [g for g in cdc2_genes if g in adata.var_names]
present_pdc = [g for g in pdc_genes if g in adata.var_names]
present_myeloid = [g for g in myeloid_genes if g in adata.var_names]

sc.tl.score_genes(adata, gene_list=present_cdc2, score_name="cdc2_score", use_raw=False)
sc.tl.score_genes(adata, gene_list=present_pdc, score_name="pdc_score", use_raw=False)
sc.tl.score_genes(adata, gene_list=present_myeloid, score_name="myeloid_score", use_raw=False)

# Summaries
cell_summary = adata.obs[["sample", "status", "gsm", "cdc2_score", "pdc_score", "myeloid_score"]].copy()
cell_summary["pdc_minus_cdc2"] = cell_summary["pdc_score"] - cell_summary["cdc2_score"]
cell_summary.to_csv(outdir / "GSE253902_dc_presence_cell_scores.csv")

sample_summary = (
    cell_summary.groupby(["sample", "status", "gsm"])
    .agg(
        n_cells=("sample", "size"),
        cdc2_score_mean=("cdc2_score", "mean"),
        pdc_score_mean=("pdc_score", "mean"),
        myeloid_score_mean=("myeloid_score", "mean"),
        pdc_minus_cdc2_mean=("pdc_minus_cdc2", "mean"),
    )
    .reset_index()
)
sample_summary.to_csv(outdir / "GSE253902_dc_presence_sample_scores.csv", index=False)

print("Merged shape:", adata.shape)
print("\nPresent cDC2 genes:", present_cdc2)
print("Present pDC genes:", present_pdc)
print("Present myeloid genes:", present_myeloid)
print("\nSample summary:")
print(sample_summary.to_string(index=False))
print("\nSaved:")
print(outdir / "GSE253902_merged_gex_raw.h5ad")
print(outdir / "GSE253902_dc_presence_cell_scores.csv")
print(outdir / "GSE253902_dc_presence_sample_scores.csv")
