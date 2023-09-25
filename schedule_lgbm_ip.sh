spatial=$1

echo spatial="${spatial}" "IP"

sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_1 9_2_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_2_2 9_2_1 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_1 9_3_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_3_2 9_3_1 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_1 9_14_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_14_2 9_14_1 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_1 9_15_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "ip" 9_15_2 9_15_1 "${spatial}"
