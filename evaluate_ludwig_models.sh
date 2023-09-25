#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=lgbm_metrics
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
spatial=$3
hyper=$4

echo "Biopsy: ${biopsy}" "Mode: ${mode}" "Spatial: ${spatial}" "Hyper: ${hyper}"

source venv/bin/activate

if [ "$spatial" != "" ]; then
  python3 evaluate_ludwig_models.py -b $1 --mode $2 --spatial $3 -s 50
else
  if [ "$hyper" == "" ]; then
    python3 evaluate_ludwig_models.py -b $1 --mode $2 -s 50
  else
    python3 evaluate_ludwig_models.py -b $1 --mode $2 --hyper -s 50
  fi
fi
