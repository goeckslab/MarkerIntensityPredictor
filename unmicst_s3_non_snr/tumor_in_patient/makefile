DATA_DIRECTORY = ../../data/tumor_s3_non_snr
DATA_DIRECTORY_PREPROCESSED = ../../data/tumor_s3_non_snr/preprocessed
SAMPLE=$(sample_id)
PREPROCESSED_DATASET = $(SAMPLE)_preprocessed_dataset.tsv
TEST_ID=$(test_id)
MARKER=$(marker)
OUTPUT_DIRECTORY = $(SAMPLE)/$(MARKER)

# Create preprocessed dataset.
"$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)":
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && python prepare_data.py "$(DATA_DIRECTORY)/$(SAMPLE).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"

# Create final configuration file.
"$(SAMPLE)_final_config.yaml": create_input_features.py base_config.yaml "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"
	python create_input_features.py "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(SAMPLE)_input_features.yaml"; cat "$(SAMPLE)_input_features.yaml" base_config.yaml > "$(SAMPLE)_final_config.yaml"

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
ludwig-experiment: "$(SAMPLE)_final_config.yaml" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig experiment --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" --config "../../$(SAMPLE)_final_config.yaml" -rs 456

ludwig-evaluate:
	python prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig evaluate --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --model_path ./results/experiment_run/model --output_directory evaluate/$(TEST_ID)

ludwig-plots:
	python create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --predicted "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/${MARKER}_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/"

clean:
	rm "$(SAMPLE)_final_config.yaml" "$(SAMPLE)_input_features.yaml"

