#!/bin/bash

test_id=$1
marker=$2

#SBATCH --nodes=1
#SBATCH --job-name=${test_id}${marker}
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu



make -f makefile ludwig-experiment test_id="${test_id}" marker="${marker}" &&
  make -f makefile ludwig-evaluate test_id="${test_id}" marker="${marker}" &&
  make -f makefile ludwig-plots test_id="${test_id}" marker="${marker}" &&
  make -f makefile clean test_id="${test_id}" marker="${marker}"
