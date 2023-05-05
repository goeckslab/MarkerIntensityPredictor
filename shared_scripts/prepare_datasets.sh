TEST_ID=$1
PATIENT_ID=$2

DATA_DIRECTORY=./data/tumor_mesmer
DATA_DIRECTORY_COMBINED=./data/tumor_mesmer/combined
DATA_DIRECTORY_COMBINED_PREPROCESSED=./data/tumor_mesmer/combined/preprocessed
DATA_DIRECTORY_PREPROCESSED=./data/tumor_mesmer/preprocessed
PREPROCESSED_DATASET=${TEST_ID}_preprocessed_dataset.tsv
COMBINED_DATASET=${PATIENT_ID}_excluded_dataset.csv
COMBINED_PREPROCESSED_DATA=${PATIENT_ID}_excluded_dataset.tsv

mkdir -p "$DATA_DIRECTORY_COMBINED" && python ./shared_scripts/combine_datasets.py -exp --dir "$DATA_DIRECTORY" --target "$TEST_ID" --output_dir "$DATA_DIRECTORY_COMBINED"
# Create preprocessed dataset for combined data
mkdir -p "$DATA_DIRECTORY_COMBINED_PREPROCESSED" && python ./shared_scripts/prepare_data.py "$DATA_DIRECTORY_COMBINED/$COMBINED_DATASET" "$DATA_DIRECTORY_COMBINED_PREPROCESSED/$COMBINED_PREPROCESSED_DATA"
# Create preprocessed dataset for single data
mkdir -p "$DATA_DIRECTORY_PREPROCESSED" && python ./shared_scripts/prepare_data.py "$DATA_DIRECTORY/$TEST_ID.csv" "$DATA_DIRECTORY_PREPROCESSED/$PREPROCESSED_DATASET"