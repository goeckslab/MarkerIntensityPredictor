#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ludwig_metrics
#SBATCH --time=5-00:00:00
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

source venv/bin/activate

if [ "$spatial" != "" ]; then
  python3 evaluate_ludwig_models.py -b $1 --mode $2 --spatial $3
else
  python3 evaluate_ludwig_models.py -b $1 --mode $2
done
