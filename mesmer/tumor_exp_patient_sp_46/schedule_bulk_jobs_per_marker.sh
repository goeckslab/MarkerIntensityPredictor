#!/bin/bash

test_id=$1
patient=$2



markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
    echo test_id="${test_id}" marker="${marker}" patient="${patient}"
    sbatch marker_job.sh "${test_id}" "${marker}" "${patient}"
done
