#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=gnn
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

biopsy=$1
replace_value=$2
mode=$3
spatial=$4
subsets=$5


python3 ./gnn/gnn.py -b "${biopsy}" -rm ${replace_value} --mode "${mode}" --spatial "${spatial}" -i 10 --subsets "${subsets}"
