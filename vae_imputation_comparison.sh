experiment=$1
parent_run_id=$2
percentage=$3
absolute=$4
prefix=$5

source venv/bin/activate
python3 imputation_run_comparison.py -e "${experiment}" -r "${prefix} ME VAE Comparison ${absolute}%" --runs "ME VAE Imputation Percentage ${percentage} Step 1" "ME VAE Imputation Percentage ${percentage} Step 2" "ME VAE Imputation Percentage ${percentage} Step 3" "ME VAE Imputation Percentage ${percentage} Step 4" "ME VAE Imputation Percentage ${percentage} Step 5" -p "${percentage}" -pr "${parent_run_id}"
python3 imputation_run_comparison.py -e "${experiment}" -r "${prefix} VAE Comparison ${absolute}%" --runs "VAE Imputation Percentage ${percentage} Step 1" "VAE Imputation Percentage ${percentage} Step 2" "VAE Imputation Percentage ${percentage} Step 3" "VAE Imputation Percentage ${percentage} Step 4" "VAE Imputation Percentage ${percentage} Step 5" -p "${percentage}" -pr "${parent_run_id}"
