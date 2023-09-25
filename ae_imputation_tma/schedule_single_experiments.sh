#!/bin/bash

replace_value=$1
experiments=$2


biopsies=('HTA14_1_bx_0' 'HTA14_1_bx_1' 'HTA14_3_bx_0' 'HTA14_4_bx_0' 'HTA14_5_bx_0' 'HTA14_8_bx_0' 'HTA14_14_bx_0' 'HTA14_17_bx_0' 'HTA14_23_bx_0' 'HTA14_27_bx_0' 'HTA14_28_bx_0' 'HTA14_28_bx_1' 'HTA14_40_bx_0' 'HTA14_40_bx_1' 'HTA14_41_bx_0' 'HTA14_45_bx_0' 'HTA14_45_bx_1')

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $experiments)
  do
      echo biopsy="${biopsy}" mode="exp" replace_value="${replace_value}" spatial="0"
      sbatch ./ae_imputation_tma/single_ae_experiment.sh "${biopsy}" "${replace_value}" "0"
  done
done