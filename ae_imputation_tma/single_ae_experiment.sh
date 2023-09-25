#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ae_tma
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
replace_value=$2
spatial=$3

source venv/bin/activate


  if [ "$spatial" != "" ]; then
      echo "spatial is set to ${spatial}"
      python3 ./ae_imputation_tma/ae.py  -b "${biopsy}" -i 10 -sp "${spatial}" -rm "${replace_value}"


  else
    echo "spatial is not set"
    python3 ./ae_imputation_tma/ae.py -b "${biopsy}" -i 10 -rm "${replace_value}"

  fi


