#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=mesmer_exp_patient_sp_184
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


test_id=$1
marker=$2
patient=$3
random_seed=$4

make -f makefile ludwig-experiment-exa test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}"