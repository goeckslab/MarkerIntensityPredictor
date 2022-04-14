experiment=$1
model=$2
file=$3
percentage=$4

python3 simple_imputation.py -e "${experiment}" -r "SI" --files "${file}" -p "$percentage"
python3 se_imputation_zero_replacement.py -e "${experiment}" --model "${model}" "VAE" --file "${file}" -r "VAE Imputation" --steps 5 -p "${percentage}"
python3 me_imputation_zero_replacement.py -e "${experiment}" --model "${model}" "ME VAE" --file "${file}" -r "ME VAE Imputation" --steps 5 -p "${percentage}"
python3 knn_imputation.py -e "${experiment}" -r "KNN Imputation" --files "${file}" -p "${percentage}"
