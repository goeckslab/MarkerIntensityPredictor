BASE_DIR=$1
TEST_ID=$2
MARKER=$3
PATIENT_ID=$4


OUTPUT_DIRECTORY=$BASE_DIR/$TEST_ID/$MARKER
DATA_DIRECTORY=./data/tumor_mesmer
DATA_DIRECTORY_COMBINED=./data/tumor_mesmer/combined
DATA_DIRECTORY_COMBINED_PREPROCESSED=./data/tumor_mesmer/combined/preprocessed
DATA_DIRECTORY_PREPROCESSED=./data/tumor_mesmer/preprocessed
PREPROCESSED_DATASET=${TEST_ID}_preprocessed_dataset.tsv
COMBINED_DATASET=${PATIENT_ID}_excluded_dataset.csv
COMBINED_PREPROCESSED_DATA=${PATIENT_ID}_excluded_dataset.tsv


# Create final configuration file.
python ./shared_scripts/create_input_features.py "$DATA_DIRECTORY_COMBINED_PREPROCESSED/$COMBINED_PREPROCESSED_DATA" "$MARKER" "${BASE_DIR}/${TEST_ID}_${MARKER}_input_features.yaml";
cat "${BASE_DIR}/${TEST_ID}_${MARKER}_input_features.yaml" ${BASE_DIR}/base_config.yaml > "${BASE_DIR}/${TEST_ID}_${MARKER}_final_config.yaml"
