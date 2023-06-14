#!/bin/bash

mode=$1
replace_value=$2
iterations=$3

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $iterations)
  do
      echo  biopsy="${biopsy}" mode="${mode}" replace_value="${replace_value}"
      sbatch ./ae_imputation/single_ae_hyper_experiment.sh "${biopsy}" "${mode}" "${replace_value}" "${noise}" "${spatial}"
  done
done