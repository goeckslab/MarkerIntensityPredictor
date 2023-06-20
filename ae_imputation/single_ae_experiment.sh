#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ae
#SBATCH --time=7-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu

biopsy=$1
mode=$2
replace_value=$3
noise=$4
spatial=$5

source venv/bin/activate


  if [ "$spatial" != "" ]; then
    echo "spatial is set"

    if [ "$noise" != "" ]; then
      python3 ./ae_imputation/ae.py -m "${mode}" -b "${biopsy}" -i 10 -sp "${spatial}" -rm "${replace_value}" -an

    else
      python3 ./ae_imputation/ae.py -m "${mode}" -b "${biopsy}" -i 10 -sp "${spatial}" -rm "${replace_value}"
    fi

  else
    echo "spatial is not set"
    if [ "$noise" != "" ]; then
      python3 ./ae_imputation/ae.py -m "${mode}" -b "${biopsy}" -i 10 -rm "${replace_value}" -an

    else
      python3 ./ae_imputation/ae.py -m "${mode}" -b "${biopsy}" -i 10 -rm "${replace_value}"
    fi
  fi


