#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=lgbm_ip
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


spatial=$1

if [ "${spatial}" == "0" ]; then
 cd mesmer/tumor_in_patient
 # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi
# check if spatial is 23
if [ "${spatial}" == "23" ]; then
  cd mesmer/tumor_in_patient_sp_23
  # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi

# check if spatial is 46
if [ "${spatial}" == "46" ]; then
  cd mesmer/tumor_in_patient_sp_46
  # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi

# check if spatial is 92
if [ "${spatial}" == "92" ]; then
  cd mesmer/tumor_in_patient_sp_92
  # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi

# check if spatial is 138
if [ "${spatial}" == "138" ]; then
  cd mesmer/tumor_in_patient_sp_138
  # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi

# check if spatial is 184
if [ "${spatial}" == "184" ]; then
  cd mesmer/tumor_in_patient_sp_184
  # perform a loop 5 times
  for i in {1..5}; do
    echo i="${i}"
    ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
    ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
    ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
    ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
    ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
    ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
    ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
    ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1
  done
fi





