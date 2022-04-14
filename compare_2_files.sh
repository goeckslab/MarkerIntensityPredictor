# Compares two specifically given files

experiment_name=$1
file_path=$2
compare_file_path=$3
seed=$4
server=$5

if [ -z "$experiment_name" ]; then
  echo "Please specify an experiment name"
  exit 1
fi

if [ -z "$file_path" ]; then
  echo "Please specify a file name"
  exit 1
fi

if [ -z "$compare_file_path" ]; then
  echo "Please specify a file name to compare."
  exit 1
fi

if [ -z "$server" ]; then
  echo "Using localhost tracking server"
  server="http://127.0.0.1:5000"
fi

if [ -z "$seed" ]; then
  echo "No seed provided. Defaulting to 1"
  seed=1
fi

file_name="$(basename -- "${file_path} .csv")"
file_name=${file_name%%.*}
compare_name="$(basename -- "${compare_file_path} .csv")"
compare_name=${compare_name%%.*}

source venv/bin/activate
echo "Starting to compare ${file_name} with ${compare_name}"
python3 reconstruction_multi_biopsy_features.py -e "${experiment_name}" --file "${file_path}" "${compare_file_path}" -r "${file_name} ${compare_name}" -s $seed


source venv/bin/activate
echo "Starting to compare ${compare_name} with ${file_name}"
python3 reconstruction_multi_biopsy_features.py -e "${experiment_name}" --file "${compare_file_path}" "${file_path}" -r "${compare_name} ${file_name}" -s $seed
