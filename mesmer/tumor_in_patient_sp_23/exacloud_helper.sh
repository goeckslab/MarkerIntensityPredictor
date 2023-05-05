#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=mesmer_out_patient
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


sample_id=$1
test_id=$2
marker=$3
patient=$4
random_seed=$5

make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}"
