# Uses only one file to train evaluate and test

experiment_name=$1
server=$2

if [ -z "$experiment_name" ]; then
  echo "Please specify an experiment name"
  exit 1
fi

if [ -z "$server" ]; then
  echo "Using localhost tracking server"
  server="http://127.0.0.1:5000"
fi

for filepath in ./data/*.csv; do
  if [[ "$filepath" == *"SARDANA.csv"* ]]; then
    continue
  fi

  for RUN in {1..5}; do
    echo "Starting run ${RUN}"
    source venv/bin/activate
    filename="$(basename -- "${filepath} .csv")"
    filename=${filename%%.*}
    python3 single_biopsy_marker_prediction.py -e "${experiment_name}" --file "${filepath}" -r "${filename}_#${RUN}"
  done
done
