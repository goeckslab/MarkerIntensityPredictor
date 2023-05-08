#!/bin/bash


mode=$1
replace_value=$2
noise=$3
spatial=$4
iterations=$5


for i in $(seq 1 $iterations)
do
  sbatch ./ae_imputation/create_ae_results.sh "${mode}" "${replace_value}" "${noise}" "${spatial}"
done