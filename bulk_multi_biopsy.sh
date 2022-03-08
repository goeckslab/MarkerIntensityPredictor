
experiment_name=$1
server=$2

if [ -z "$experiment_name" ]
then
      echo "Please specify an experiment name"
	  exit 1
fi

if [ -z "$server" ]
then
    echo "Using localhost tracking server"
	  server="http://127.0.0.1:5000"
fi

for filename in ./data/*.csv;
do
  if [[ "$filename" == "SARDANA.csv" ]]
  then
    continue
  fi

  for comparefile in ./data/*csv;
  do

    if [[ "$comparefile" == "SARDANA.csv" ]]
    then
      continue
    fi

     if [[ "$comparefile" == "$filename" ]]
    then
      continue
    fi

    for RUN in {1..10}
    do
	    echo "Starting run ${RUN} comparing ${filename} with ${comparefile}"
	    source venv/bin/activate
	    python3 multi_biopsy_marker_prediction.py -e "${experiment_name}" --file "${filename}" "${comparefile}" -r "${RUN}"
    done
  done
done
