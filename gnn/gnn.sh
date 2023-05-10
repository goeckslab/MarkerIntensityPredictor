#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=gnn
#SBATCH --time=5-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

folder=$1
replace_value=$2

python3 ./gnn/gnn.py -f "${folder}" -rm ${replace_value} -i 10
