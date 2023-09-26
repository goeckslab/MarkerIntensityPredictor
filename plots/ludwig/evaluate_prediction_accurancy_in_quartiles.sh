#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=gbm_quartiles
#SBATCH --time=5-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kirchgae@ohsu.edu

spatial=$1

python3 ./plots/ludwig/evaluate_prediction_accurancy_in_quartiles.py --sp ${spatial}