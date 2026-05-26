import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
clinical_file = project_dir / "data" / "clinical" / "human_immune_health_atlas_metadata_clinical_labs.csv"

cd8_l2 = pd.read_csv(results_dir / "cd8_L2_with_samplemeta.csv")
cd8_l3 = pd.read_csv(results_dir / "cd8_L3_with_samplemeta.csv")
clinical = pd.read_csv(clinical_file)

keep_cols = [
    "sample.sampleKitGuid",
    "subject.subjectGuid",
    "subject.biologicalSex",
    "subject.ageAtFirstDraw",
    "subject.race",
    "subject.ethnicity",
    "sample.drawYear",
    "sample.subjectAgeAtDraw",
    "cmv.igg_serology_interpretation",
    "am.bmi",
    "infl.hs_crp",
    "infl.esr",
    "chem.glucose",
    "lip.cholesterol_hdl",
    "lip.cholesterol_ldl",
    "lip.cholesterol_non_hdl",
    "lip.cholesterol_total",
    "lip.chlesterol_hdl_ratio",
    "lip.triglycerides",
]
keep_cols = [c for c in keep_cols if c in clinical.columns]
clinical = clinical[keep_cols].copy()
clinical = clinical.drop_duplicates(subset=["sample.sampleKitGuid"]).copy()

cd8_l2m = cd8_l2.merge(clinical, on="sample.sampleKitGuid", how="left", suffixes=("", "_clinical"))
cd8_l3m = cd8_l3.merge(clinical, on="sample.sampleKitGuid", how="left", suffixes=("", "_clinical"))

cd8_l2m.to_csv(results_dir / "cd8_L2_with_clinical_by_sample.csv", index=False)
cd8_l3m.to_csv(results_dir / "cd8_L3_with_clinical_by_sample.csv", index=False)

print("Done.")
print(results_dir / "cd8_L2_with_clinical_by_sample.csv")
print(results_dir / "cd8_L3_with_clinical_by_sample.csv")
print("\nL2 shape:", cd8_l2m.shape)
print("L3 shape:", cd8_l3m.shape)
