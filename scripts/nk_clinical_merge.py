import pandas as pd
from pathlib import Path

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
clinical_file = project_dir / "data" / "clinical" / "human_immune_health_atlas_metadata_clinical_labs.csv"

nk_l2 = pd.read_csv(results_dir / "nk_subject_level_AIFI_L2_composition_clean.csv")
nk_l3 = pd.read_csv(results_dir / "nk_subject_level_AIFI_L3_composition_clean.csv")
clinical = pd.read_csv(clinical_file)

# Keep a focused clinical subset
keep_cols = [
    'subject.subjectGuid',
    'sample.sampleKitGuid',
    'subject.biologicalSex',
    'subject.ageAtFirstDraw',
    'subject.race',
    'subject.ethnicity',
    'sample.drawYear',
    'sample.subjectAgeAtDraw',
    'cmv.igg_serology_interpretation',
    'am.bmi',
    'infl.hs_crp',
    'infl.esr',
    'chem.glucose',
    'lip.cholesterol_hdl',
    'lip.cholesterol_ldl',
    'lip.cholesterol_non_hdl',
    'lip.cholesterol_total',
    'lip.chlesterol_hdl_ratio',
    'lip.triglycerides',
]
keep_cols = [c for c in keep_cols if c in clinical.columns]
clinical = clinical[keep_cols].copy()

# Merge on subject first; upgrade to sample.sampleKitGuid later if present in NK outputs
nk_l2m = nk_l2.merge(clinical, on='subject.subjectGuid', how='left')
nk_l3m = nk_l3.merge(clinical, on='subject.subjectGuid', how='left')

nk_l2m.to_csv(results_dir / 'nk_L2_with_clinical.csv', index=False)
nk_l3m.to_csv(results_dir / 'nk_L3_with_clinical.csv', index=False)

print("Done.")
print(results_dir / 'nk_L2_with_clinical.csv')
print(results_dir / 'nk_L3_with_clinical.csv')
print("\nNK L2 merged shape:", nk_l2m.shape)
print("NK L3 merged shape:", nk_l3m.shape)
