experiment=$1
parent_run_id=$2
percentage=$3
absolute=$4
prefix=$5

source venv/bin/activate
python3 imputation_run_comparison.py -e "${experiment}" -r "${prefix} All Tools Comparison ${absolute}%" --runs "SI Imputation Percentage ${percentage}" "KNN Imputation Percentage ${percentage}" "VAE Imputation Percentage ${percentage} Step 1 " "ME VAE Imputation Percentage ${percentage} Step 1 " -p "${percentage}" -pr "${parent_run_id}"
