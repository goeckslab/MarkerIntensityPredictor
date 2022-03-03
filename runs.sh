
run_name=$1
file_name=$2

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

for RUN in {1..15}
do
	echo "Starting run ${RUN}"
	source venv/bin/activate
	python3 src/marker_intensity_predictor.py -e VAE_AE_Comparison --file ${file_name} -t http://127.0.0.1:5000 -r ${run_name}_"#${RUN}"
done
