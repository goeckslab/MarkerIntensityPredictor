#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=gnn_prepare_data
#SBATCH --time=5-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

biopsy=$1
mode=$2
spatial=$3

python3 gnn/prepare_data.py -b "${biopsy}" --mode "${mode}" --spatial "${spatial}"