spatial=$1

echo spatial="${spatial}" "IP"

if [ "${spatial}" == "0" ]; then
  cd mesmer/tumor_in_patient
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi

if [ "${spatial}" == "23" ]; then
  cd mesmer/tumor_in_patient_sp_23
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi

if [ "${spatial}" == "46" ]; then
  cd mesmer/tumor_in_patient_sp_46
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi

if [ "${spatial}" == "92" ]; then
  cd mesmer/tumor_in_patient_sp_92
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi

if [ "${spatial}" == "138" ]; then
  cd mesmer/tumor_in_patient_sp_138
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi

if [ "${spatial}" == "184" ]; then
  cd mesmer/tumor_in_patient_sp_184
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2
  sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1
fi
