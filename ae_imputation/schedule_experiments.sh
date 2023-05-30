#!/bin/bash


mode=$1
replace_value=$2
noise=$3
spatial=$4
iterations=$5
hyper=$6


for i in $(seq 1 $iterations)
do
  if [ "$hyper" == "" ]; then
    echo mode="${mode}" replace_value="${replace_value}" noise="${noise}" spatial="${spatial}"
    sbatch ./ae_imputation/create_ae_results.sh "${mode}" "${replace_value}" "${noise}" "${spatial}"
  else
    echo mode="${mode}" replace_value="${replace_value}" noise="${noise}" spatial="${spatial}" hyper="${hyper}"
    sbatch ./ae_imputation/create_ae_hp_results.sh "${mode}" "${replace_value}" "${noise}" "${spatial}"
  fi


done