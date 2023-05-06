#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=mesmer_in_patient_sp_46
#SBATCH --time=0-48:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/${marker}_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kirchgae@ohsu.edus

sample_id=$1
test_id=$2
patient=$3
iterations=$4

echo ${iterations}

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
  for i in $(seq 1 $iterations)
  do
    random_seed=$RANDOM
    echo "${marker}"

    echo sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient="${patient}" random_seed="${random_seed}"
    sbatch exacloud_helper.sh "${sample_id}" "${test_id}" "${marker}" "${patient}" "${random_seed}"
  done

done
