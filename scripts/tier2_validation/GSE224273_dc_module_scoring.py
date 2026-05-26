from pathlib import Path
import pandas as pd
import scanpy as sc
from scipy import io

base = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/data/tier2_validation/GSE224273/extracted")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

samples = {
    "Sample1":  {"gsm": "GSM7018579", "status": "Symptomatic"},
    "Sample2":  {"gsm": "GSM7018580", "status": "Asymptomatic"},
    "Sample3":  {"gsm": "GSM7018581", "status": "Asymptomatic"},
    "Sample3A": {"gsm": "GSM7018582", "status": "Asymptomatic"},
    "Sample4":  {"gsm": "GSM7018583", "status": "Asymptomatic"},
    "Sample5":  {"gsm": "GSM7018584", "status": "Asymptomatic"},
    "Sample1G": {"gsm": "GSM7018585", "status": "Symptomatic"},
}

adatas = []

for sample, meta in samples.items():
    gsm = meta["gsm"]
    genes_f = base / f"{gsm}_{sample}_genes.tsv.gz"
    barcodes_f = base / f"{gsm}_{sample}_barcodes.tsv.gz"
    matrix_f = base / f"{gsm}_{sample}_matrix.mtx.gz"

    genes = pd.read_csv(genes_f, sep="\t", header=None)
    if genes.shape[1] == 2:
        genes.columns = ["ensembl", "gene"]
    else:
        genes = genes.iloc[:, :2]
        genes.columns = ["ensembl", "gene"]

    barcodes = pd.read_csv(barcodes_f, sep="\t", header=None, names=["barcode"])
    X = io.mmread(matrix_f).tocsr().T

    var = genes.copy()
    var.index = var["gene"].astype(str)
    obs = pd.DataFrame(index=barcodes["barcode"].astype(str))
    obs["sample"] = sample
    obs["gsm"] = gsm
    obs["status"] = meta["status"]

    ad = sc.AnnData(X=X, obs=obs, var=var)
    ad.var_names_make_unique()
    adatas.append(ad)

adata = sc.concat(adatas, join="outer", index_unique="-")
adata.write(outdir / "GSE224273_merged_rna_raw.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cdc2_genes = ["HLA-DRA", "HLA-DRB1", "CD74", "FCER1A", "CD1C", "CLEC10A"]
pdc_genes = ["GZMB", "TCF4", "JCHAIN", "IRF7"]
myeloid_genes = ["LST1", "CTSS", "FCN1", "TYMP", "S100A8", "S100A9"]

present_cdc2 = [g for g in cdc2_genes if g in adata.var_names]
present_pdc = [g for g in pdc_genes if g in adata.var_names]
present_myeloid = [g for g in myeloid_genes if g in adata.var_names]

sc.tl.score_genes(adata, present_cdc2, score_name="cdc2_score", use_raw=False)
sc.tl.score_genes(adata, present_pdc, score_name="pdc_score", use_raw=False)
sc.tl.score_genes(adata, present_myeloid, score_name="myeloid_score", use_raw=False)

cell_df = adata.obs[["sample", "gsm", "status", "cdc2_score", "pdc_score", "myeloid_score"]].copy()
cell_df["pdc_minus_cdc2"] = cell_df["pdc_score"] - cell_df["cdc2_score"]
cell_df.to_csv(outdir / "GSE224273_dc_cell_scores.csv")

sample_df = (
    cell_df.groupby(["sample", "gsm", "status"], observed=True)
    .agg(
        n_cells=("sample", "size"),
        cdc2_score_mean=("cdc2_score", "mean"),
        pdc_score_mean=("pdc_score", "mean"),
        myeloid_score_mean=("myeloid_score", "mean"),
        pdc_minus_cdc2_mean=("pdc_minus_cdc2", "mean"),
    )
    .reset_index()
)

sample_df.to_csv(outdir / "GSE224273_dc_sample_scores.csv", index=False)

print("Merged shape:", adata.shape)
print("Present cDC2 genes:", present_cdc2)
print("Present pDC genes:", present_pdc)
print("Present myeloid genes:", present_myeloid)
print("\nSample summary:")
print(sample_df.to_string(index=False))
print("\nSaved:")
print(outdir / "GSE224273_merged_rna_raw.h5ad")
print(outdir / "GSE224273_dc_cell_scores.csv")
print(outdir / "GSE224273_dc_sample_scores.csv")
