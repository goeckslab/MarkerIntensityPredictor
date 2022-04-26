#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=Missing_Marker_Imputation
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

source venv/bin/activate
python3 remove_marker_from_training.py --folder "$1" --exclude "$2" -nml
