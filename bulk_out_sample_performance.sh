# Compares all sample in a folder with each other

experiment_name=$1
seed=$2
server=$3

if [ -z "$experiment_name" ]; then
  echo "Please specify an experiment name"
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

for filepath in ./data/*.csv; do
  if [[ "$filepath" == *"SARDANA.csv"* ]]; then
    continue
  fi

  for comparepath in ./data/*csv; do
    if [[ "$comparepath" == *"SARDANA.csv"* ]]; then
      continue
    fi

    if [[ "$comparepath" == "$filepath" ]]; then
      continue
    fi

    source venv/bin/activate
    filename="$(basename -- "${filepath} .csv")"
    filename=${filename%%.*}

    comparename="$(basename -- "${comparepath} .csv")"
    comparename=${comparename%%.*}
    python3 reconstruction_multi_biopsy_features.py -e "${experiment_name}" --file "${filepath}" "${comparepath}" -r "${filename}_${comparename}" -s $seed


  done
done
