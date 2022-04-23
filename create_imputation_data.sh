experiment=$1
file=$2
percentage=$3

python3 single_feature_imputation.py -e "${experiment}" --model "${experiment}" "Model" --file "${file}" --steps 5 -p "${percentage}" -r "Single Feature Imputation"
python3 multi_feature_imputation.py -e "${experiment}" --model "${experiment}" "Model" --file "${file}" --steps 5 -p "${percentage}" -r "Multi Feature Imputation"
