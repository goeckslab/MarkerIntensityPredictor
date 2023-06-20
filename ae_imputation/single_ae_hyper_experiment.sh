#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ae_hp
#SBATCH --time=9-00:00:00
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
spatial=$4

source venv/bin/activate
echo "Running hp"

  if [ "$spatial" != "" ]; then
    echo "spatial is set"
    python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -sp "${spatial}" -rm "${replace_value}" -hp

  else
    echo "spatial is not set"
    python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -rm "${replace_value}" -hp

  fi


