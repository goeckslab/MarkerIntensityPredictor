#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=KNN_Distance_Imputation
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

source venv/bin/activate
python3 distance_based_neighborhood_imputation.py --folder "$1" --exclude "$2" -d "$3"
