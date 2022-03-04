
experiment_name=$1
run_name=$2
file_name=$3
server=$4

if [ -z "$experiment_name" ]
then
      echo "Please specify an experiment name"
	  exit 1
fi


if [ -z "$run_name" ]
then
      echo "Please specify a run name"
	  exit 1
fi

if [ -z "$file_name" ]
then
      echo "Please specify a file to use"
	  exit 1
fi

if [ -z "$server" ]
then
      echo "Using localhost tracking server"
	  $server="http://127.0.0.1:5000"
fi

for RUN in {1..15}
do
	echo "Starting run ${RUN}"
	source venv/bin/activate
	python3 src/marker_intensity_predictor.py -e ${experiment_name} --file ${file_name} -t ${server} -r ${run_name}_"#${RUN}"
done
