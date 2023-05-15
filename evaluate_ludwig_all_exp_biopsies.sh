# Always add a trailing slash to the folder paths

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

for biopsy in "${biopsies[@]}"; do
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp"
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp" 23
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp" 46
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp" 92
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp" 138
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "exp" 184
done