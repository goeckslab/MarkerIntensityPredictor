#!/bin/bash

sample_id=$1
test_id=$2
iterations=$3

echo ${iterations}

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
  for i in $(seq 1 $iterations)
  do
    echo "${marker}"

    echo sample_id="${sample_id}" test_id="${test_id}" marker="${marker}"
    sbatch exacloud_helper.sh "${sample_id}" "${test_id}" "${marker}"
  done

done
