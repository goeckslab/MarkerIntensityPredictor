spatial=$1

echo spatial="${spatial}" "EXP"

sbatch ./schedule_lgbm_biopsy.sh "exp" 9_2_1 9_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_2_2 9_2 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_3_1 9_3 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_3_2 9_3 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_14_1 9_14 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_14_2 9_14 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_15_1 9_15 "${spatial}"
sbatch ./schedule_lgbm_biopsy.sh "exp" 9_15_2 9_15 "${spatial}"
