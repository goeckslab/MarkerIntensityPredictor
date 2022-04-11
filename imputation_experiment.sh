model=$1
file=$2
run_name=$3
percentage=$4

python3 imputation_random_mean.py -e "${model}" -m "${model}" "Model" --file "${file}" --run "${run_name}" -p "${percentage}" --steps 1
python3 imputation_random_mean.py -e "${model}" -m "${model}" "Model" --file "${file}" --run "${run_name}" -p "${percentage}" --steps 2
python3 imputation_random_mean.py -e "${model}" -m "${model}" "Model" --file "${file}" --run "${run_name}" -p "${percentage}" --steps 3
python3 imputation_random_mean.py -e "${model}" -m "${model}" "Model" --file "${file}" --run "${run_name}" -p "${percentage}" --steps 4
python3 imputation_random_mean.py -e "${model}" -m "${model}" "Model" --file "${file}" --run "${run_name}" -p "${percentage}" --steps 5


python3 imputation_run_comparison.py -e "${model}" --runs "${run_name} Percentage ${percentage} Steps 1" "${run_name} Percentage ${percentage} Steps 2" "${run_name} Percentage ${percentage} Steps 3" "${run_name} Percentage ${percentage} Steps 4"  "${run_name} Percentage ${percentage} Steps 5" -r "${run_name} Percentage ${percentage} Summary"
