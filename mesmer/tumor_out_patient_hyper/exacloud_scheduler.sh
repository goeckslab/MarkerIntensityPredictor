#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unmicst_snr_out_patient_hyper
#SBATCH --time=0-48:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/${marker}_slurm.%N.%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kirchgae@ohsu.edus

test_id=$1

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
#markers=('pRB')
for marker in "${markers[@]}"; do
  echo "${marker}"
  echo test_id="${test_id}" marker="${marker}"

  sbatch exacloud_executor.sh "${test_id}" "${marker}"
done
