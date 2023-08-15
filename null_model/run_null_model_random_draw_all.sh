#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=rd_all
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

python3 null_model/null_model_random_draw_all.py -ex 100