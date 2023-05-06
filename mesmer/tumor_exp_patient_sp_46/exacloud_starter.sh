#!/bin/bash

test_id=$1
patient=$2
iterations=$3

echo ${iterations}

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
  for i in $(seq 1 $iterations)
  do
    random_seed=$RANDOM
    echo "${marker}"

    echo test_id="${test_id}" marker="${marker}" patient="${patient}" random_seed="${random_seed}"
    sbatch exacloud_helper.sh "${test_id}" "${marker}" "${patient}" "${random_seed}"
  done

done
