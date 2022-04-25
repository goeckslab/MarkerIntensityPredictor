#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH -p gpu --gres gpu:v100:1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu
#SBATCH --mem 64G

source venv/bin/activate
python3 remove_marker_from_training.sh --data