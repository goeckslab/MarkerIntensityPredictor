#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=auto_encoder
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edu


mode=$1
replace_value=$2
noise=$3
spatial=$4
iterations=$5


source venv/bin/activate

for i in $(seq 1 $iterations)
do
  if [ "$spatial" != "" ]; then
    echo "spatial is set"

    if [ "$noise" != "" ]; then
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_2 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_1 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_2 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_1 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_2 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_1 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_2 -i 10 -sp "${spatial}" -rm "${replace_value}" -an
    else
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_2 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_1 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_2 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_1 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_2 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_1 -i 10 -sp "${spatial}" -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_2 -i 10 -sp "${spatial}" -rm "${replace_value}"
    fi

  else
    echo "spatial is not set"
    if [ "$noise" != "" ]; then
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_2 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_1 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_2 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_1 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_2 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_1 -i 10 -rm "${replace_value}" -an
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_2 -i 10 -rm "${replace_value}" -an
    else
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_1 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_2_2 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_1 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_3_2 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_1 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_14_2 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_1 -i 10 -rm "${replace_value}"
      python3 ./ae_imputation/ae.py -m "${mode}" -b 9_15_2 -i 10 -rm "${replace_value}"
    fi
  fi
done


