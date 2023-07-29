#!/bin/bash

mode=$1
replace_value=$2
experiments=$3
spatial=$4

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $experiments)
  do
      echo biopsy="${biopsy}" mode="${mode}" replace_value="${replace_value}" spatial="${spatial}" biopsy="${biopsy}"
      ./ae_imputation/single_ae_experiment.sh "${biopsy}" "${mode}" "${replace_value}" "${spatial}"
  done
done