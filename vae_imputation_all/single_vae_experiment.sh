#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=vae_all
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

biopsy=$1
mode=$2
subsets=$3

source venv/bin/activate
python3 ./vae_imputation_all/vae.py -m "${mode}" -b "${biopsy}" -i 10 -s "${subsets}"


