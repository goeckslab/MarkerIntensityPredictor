#!/bin/bash


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
