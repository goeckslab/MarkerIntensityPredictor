#!/bin/bash

sample_id=$1
test_id=$2
patient=$3



markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
    random_seed=$RANDOM

    echo sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" patient="${patient}"
    sbatch marker_job.sh "${sample_id}" "${test_id}" "${marker}" "${patient}"
done
