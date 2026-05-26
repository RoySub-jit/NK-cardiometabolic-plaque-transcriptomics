#!/bin/bash
#SBATCH -J gse198339_build
#SBATCH -p standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /pub/subhajr/projects/heat_aging_human_reanalysis/logs/gse198339_build_%j.out
#SBATCH -e /pub/subhajr/projects/heat_aging_human_reanalysis/logs/gse198339_build_%j.err

set -euo pipefail

cd /pub/subhajr/projects/heat_aging_human_reanalysis
source ~/software/miniforge3/etc/profile.d/conda.sh
conda activate /data/homezvol2/subhajr/envs/scanpy-env

python /pub/subhajr/projects/heat_aging_human_reanalysis/scripts/tier2_validation/build_GSE198339_singlecell_object.py
