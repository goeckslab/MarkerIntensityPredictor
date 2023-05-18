#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ip_92
#SBATCH --time=5-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --requeue
#SBATCH --mail-user=kirchgae@ohsu.edu


sample_id=$1
test_id=$2
marker=$3
patient=$4
random_seed=$RANDOM

echo $random_seed

make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}" ;
make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}" ;
make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}" ;
make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}" ;
make -f makefile ludwig-experiment-exa sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed="${random_seed}"

