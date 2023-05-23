#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=en_exp
#SBATCH --time=10-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --requeue
#SBATCH --mail-user=kirchgae@ohsu.edu

python3 en/elastic_net.py -b 9_2_1 --mode exp
python3 en/elastic_net.py -b 9_2_2 --mode exp
python3 en/elastic_net.py -b 9_3_1 --mode exp
python3 en/elastic_net.py -b 9_3_2 --mode exp
python3 en/elastic_net.py -b 9_14_1 --mode exp
python3 en/elastic_net.py -b 9_14_2 --mode exp
python3 en/elastic_net.py -b 9_15_1 --mode exp
python3 en/elastic_net.py -b 9_15_2 --mode exp
