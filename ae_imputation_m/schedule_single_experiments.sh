#!/bin/bash

mode=$1
replace_value=$2
# How many times should each experiment be repeated?
experiments=$3
# the spatial radius
spatial=$4
# How many times should the test data be subsampled
subsets=$5

# if subsets is not set, set it to 1
if [ "$subsets" == "" ];
then
  echo "Subsets was not set. Setting it to 1."
  subsets=1
fi


if [ "$mode" == "" ];
then
  echo "Mode is not set. Setting it to IP"
  mode="ip"
fi

if [ "$replace_value" == "" ];
then
  echo "Replace Value is not set. Setting it to mean"
  replace_value="mean"
fi

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $experiments)
  do
      echo biopsy="${biopsy}" mode="${mode}" replace_value="${replace_value}" subsets="${subsets}" spatial="${spatial}"
      sbatch ./ae_imputation_m/single_ae_experiment.sh "${biopsy}" "${mode}" "${replace_value}" "${spatial}" "${subsets}"
  done
done