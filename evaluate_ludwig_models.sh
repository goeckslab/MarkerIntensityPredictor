#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ludwig_model_evaluation
#SBATCH --time=0-48:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


biopsy=$1
dataset=$2

source venv/bin/activate

python3 evaluate_ludwig_models.py -b $1 -d $2