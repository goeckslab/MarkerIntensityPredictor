#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=exp_patient
#SBATCH --time=5-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --mem=16000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

biopsy=$1
mode=$2

python3 gnn/prepare_data.py -b "${biopsy}" -m "${mode}"