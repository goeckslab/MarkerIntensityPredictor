#!/bin/bash

test_id=$1
patient=$2


markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
    random_seed=$RANDOM
    echo test_id="${test_id}" marker="${marker}" patient="${patient}" random_seed="${random_seed}"
    sbatch exacloud_helper.sh "${test_id}" "${marker}" "${patient}" "${random_seed}"
done
