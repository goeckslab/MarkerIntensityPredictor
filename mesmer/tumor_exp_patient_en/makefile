DATA_DIRECTORY = ../../data/tumor_mesmer
DATA_DIRECTORY_PREPROCESSED = ../../data/tumor_mesmer/preprocessed
DATA_DIRECTORY_COMBINED = ../../data/tumor_mesmer/op_combined
DATA_DIRECTORY_COMBINED_PREPROCESSED = ../../data/tumor_mesmer/op_combined/preprocessed
TEST_ID=$(test_id)
PREPROCESSED_TEST_DATASET = $(TEST_ID)_preprocessed_dataset.tsv
COMBINED_DATASET = $(TEST_ID)_excluded_dataset.csv
COMBINED_PROCESSED_DATASET = $(TEST_ID)_excluded_dataset.tsv
MARKER=$(marker)
OUTPUT_DIRECTORY = $(TEST_ID)/$(TEST_ID)/$(MARKER)

# Create preprocessed dataset.
prepare_data:
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && \
	mkdir -p "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)" && \
	python prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)" && \
	python prepare_data.py "$(DATA_DIRECTORY_COMBINED)/$(COMBINED_DATASET)" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PROCESSED_DATASET)"


# Create combined dataset
combine_data_sets:
	mkdir -p "$(DATA_DIRECTORY_COMBINED)" && python combine_datasets.py --dir "$(DATA_DIRECTORY)" --target "$(TEST_ID)" --output_dir "$(DATA_DIRECTORY_COMBINED)"

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
run_en: combine_data_sets prepare_data
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PROCESSED_DATASET)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)"

plots:
	python ../../shared_scripts/create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)" --predicted "$(OUTPUT_DIRECTORY)/$(MARKER)_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)"

run-en-exa:
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PROCESSED_DATASET)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)"
