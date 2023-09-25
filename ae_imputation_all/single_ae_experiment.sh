#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ae_all
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
repetitions=$5

# if repetitions is not set, set it to 1
if [ "$repetitions" == "" ];
then
  echo "Repetitions was not set. Setting it to 1."
  exit 1
fi

source venv/bin/activate
if [ "$spatial" != "" ]; then
  echo "spatial is set"
  python3 ./ae_imputation_all/ae.py -m "${mode}" -b "${biopsy}" -i 10 -sp "${spatial}" -rm "${replace_value}" -r "${repetitions}"

else
  echo "spatial is not set"
  python3 ./ae_imputation_all/ae.py -m "${mode}" -b "${biopsy}" -i 10 -rm "${replace_value}" -r "${repetitions}"
fi


