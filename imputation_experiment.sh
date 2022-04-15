experiment=$1
run_name=$2
file=$3
percentage=$4




python3 se_imputation_zero_replacement.py -e "${experiment}" --model "${experiment}" "${run_name}" "VAE" --file "${file}" -r "VAE Imputation" --steps 5 -p "${percentage}"
python3 me_imputation_zero_replacement.py -e "${experiment}" --model "${experiment}" "${run_name}" "ME VAE" --file "${file}" -r "ME VAE Imputation" --steps 5 -p "${percentage}"
python3 simple_imputation.py -e "${experiment}" -r "SI" --files "${file}" -p "$percentage"
python3 knn_imputation.py -e "${experiment}" -r "KNN Imputation" --files "${file}" -p "${percentage}"
