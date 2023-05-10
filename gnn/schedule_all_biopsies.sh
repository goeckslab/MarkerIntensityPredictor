
#!/bin/bash

mode=$1
spatial=$2
replace_value=$3
biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

base_path="./gnn/data/"

for biopsy in "${biopsies[@]}"; do
    data_path=$base_path$mode/$biopsy/$spatial
    ./gnn/gnn.sh "${data_path}" "${replace_value}"

done