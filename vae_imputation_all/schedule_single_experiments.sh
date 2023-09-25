#!/bin/bash

mode=$1
experiments=$2
subsets=$3

# if repetitions == "" set experiments to 1
if [ "$experiments" == "" ]; then
  experiments=1
fi

if [ "$subsets" == "" ]; then
  subsets=1
fi


biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $experiments)
  do
      echo biopsy="${biopsy}" mode="${mode}"  biopsy="${biopsy}" subsets="${subsets}"
      ./vae_imputation_all/single_vae_experiment.sh "${biopsy}" "${mode}" "${subsets}"
  done
done