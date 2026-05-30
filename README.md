# NK/cytotoxic-associated immune programs across cardiometabolic and vascular inflammatory contexts

Repository supporting the manuscript:

**Integrative single-cell and plaque transcriptomic analyses identify recurring NK/cytotoxic-associated immune programs across cardiometabolic and vascular inflammatory contexts**

## Overview

This repository contains analysis scripts, verified derived outputs, figure-generation materials, supplementary tables, and reproducibility audit files supporting an integrative computational reanalysis of publicly available transcriptomic datasets.

The study evaluates NK/cytotoxic-associated immune features across:

- healthy donor immune single-cell data linked to non-HDL cholesterol;
- an external PBMC disease-context cohort;
- carotid plaque single-cell transcriptomic comparisons; and
- intraplaque hemorrhage-stratified bulk plaque transcriptomics.

## Public datasets

- Allen Human Immune Health Atlas: healthy donor discovery analysis
- GSE198339: external PBMC disease-context analysis
- GSE224273: carotid plaque single-cell comparison
- GSE163154: intraplaque hemorrhage-stratified plaque analysis

Raw public datasets are not redistributed in this repository.

## Verified GSE198339 reanalysis

The external PBMC disease-context component was audited and reproduced using the official processed GSE198339 dataset comprising 9,368 PBMCs from eight male participants living with HIV, four with and four without atherosclerosis. Participant-level non-HDL cholesterol was calculated as total cholesterol minus HDL cholesterol using the official accompanying metadata.

The verified reanalysis supported positive associations of non-HDL cholesterol with:

- cytotoxic core score;
- NKG7 expression within annotated NK cells; and
- NK resting-cell proportion.

The GZMK-like composite score showed a positive non-significant trend and is interpreted accordingly in the manuscript.

## Verified GSE224273 Figure 3 revision

Figure 3 was revised to retain plaque-derived analyses only. The corrected panels are based on GSE224273 plaque single-cell data and do not include PBMC cluster-comparison, PBMC pathway, or unsupported cross-dataset pathway/FDR values. The plaque analysis identified a positive trend for the GZMK-like score in asymptomatic relative to symptomatic plaques, with broader module-level findings interpreted as exploratory.

## Repository structure

- `scripts/final/GSE198339_verified/`: verified external PBMC disease-context scripts
- `scripts/first_dataset/`: Allen healthy-donor discovery scripts used for manuscript outputs
- `scripts/tier2_validation/`: retained plaque single-cell provenance scripts
- `scripts/final/GSE224273_verified/`: verified plaque-only Figure 3 generation scripts
- `scripts/tier3_plaque_validation/`: IPH bulk plaque scripts retained for manuscript outputs
- `results_final_main/`: derived numerical results supporting the final figure package
- `figures_final_main/`: verified corrected figure panels generated from the final analysis workflow
- `supplementary_figures/`: supplementary figure outputs
- `supplementary_tables/`: final verified supplementary workbook
- `reproducibility_audit/`: audit records supporting corrected reanalysis

## Data and code availability

Public transcriptomic datasets analyzed in this study are available from the Allen Human Immune Health Atlas and NCBI Gene Expression Omnibus under accession numbers GSE198339, GSE224273, and GSE163154. This repository provides analysis scripts, verified derived outputs, figure-generation materials, and supplementary files used for the manuscript. Large intermediate objects, computational environments, and raw public dataset downloads are intentionally not redistributed.
