
experiment_name=$1
file_name=$2
compare_file_name=$3
seed=$4
server=$5

if [ -z "$experiment_name" ]
then
      echo "Please specify an experiment name"
	  exit 1
fi

if [ -z "$file_name" ]
then
      echo "Please specify a file name"
	  exit 1
fi

if [ -z "$compare_file_name" ]
then
      echo "Please specify a file name to compare."
	  exit 1
fi

if [ -z "$server" ]
then
    echo "Using localhost tracking server"
	  server="http://127.0.0.1:5000"
fi

if [ -z "$seed" ]
then
      echo "No seed provided. Defaulting to 1"
      seed=1
fi

 for RUN in {1..5}
    do
	    echo "Starting run ${RUN} comparing ${file_name} with ${compare_file_name}"
	    source venv/bin/activate
	    python3 multi_biopsy_marker_prediction.py -e "${experiment_name}" --file "${file_name}" "${compare_file_name}" -r "${RUN}" -s $seed
    done

for RUN in {1..5}
  do
	  echo "Starting run ${RUN} comparing ${compare_file_name} with ${file_name}"
	  source venv/bin/activate
	  python3 multi_biopsy_marker_prediction.py -e "${experiment_name}" --file "${compare_file_name}" "${file_name}" -r "${RUN}" -s $seed
  done