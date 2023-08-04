#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=lgbm_exp
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


mode=$1
biopsy=$2



if [ "${mode}" == "exp" ];
then
  patient=$3
  echo mode="${mode}" biopsy="${biopsy}" patient="${patient}"
  ./evaluate_all_marker_out_patient.sh $biopsy $patient
  ./evaluate_all_marker_out_patient.sh $biopsy $patient
  ./evaluate_all_marker_out_patient.sh $biopsy $patient
  ./evaluate_all_marker_out_patient.sh $biopsy $patient
  ./evaluate_all_marker_out_patient.sh $biopsy $patient
else
  test_biopsy=$3
  echo mode="${mode}" biopsy="${biopsy}" test_biopsy="${test_biopsy}"
  ./evaluate_all_marker_in_patient.sh $biopsy $test_biopsy
  ./evaluate_all_marker_in_patient.sh $biopsy $test_biopsy
  ./evaluate_all_marker_in_patient.sh $biopsy $test_biopsy
  ./evaluate_all_marker_in_patient.sh $biopsy $test_biopsy
  ./evaluate_all_marker_in_patient.sh $biopsy $test_biopsy
fi