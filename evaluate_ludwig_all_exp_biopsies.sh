# Always add a trailing slash to the folder paths


# the folder where ALL biopsy models are stored
model_folder=$1
# the folder where the preprocessed data is stored
data_folder=$2

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

for biopsy in "${biopsies[@]}"; do
  biopsy_folder="${model_folder}${biopsy}/"
  biopsy_data="${data_folder}${biopsy}_preprocessed_dataset.tsv"
  sbatch ./evaluate_ludwig_models.sh "${biopsy_folder}" "${biopsy_data}"
done