# Always add a trailing slash to the folder paths


biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

for biopsy in "${biopsies[@]}"; do
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip"
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip" 23
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip" 46
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip" 92
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip" 138
  sbatch ./evaluate_ludwig_models.sh "${biopsy}" "ip" 184
done