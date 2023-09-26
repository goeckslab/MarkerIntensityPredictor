DATA_DIRECTORY = ../../data/tumor_mesmer
DATA_DIRECTORY_PREPROCESSED = ../../data/tumor_mesmer/preprocessed
SAMPLE=$(sample_id)
PREPROCESSED_TRAIN_DATASET = $(SAMPLE)_preprocessed_dataset.tsv
TEST=$(test_id)
PREPROCESSED_TEST_DATASET = $(TEST)_preprocessed_dataset.tsv
MARKER=$(marker)
OUTPUT_DIRECTORY = $(SAMPLE)/$(TEST)/$(MARKER)

# Create preprocessed dataset.
prepare_data:
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && \
	python prepare_data.py "$(DATA_DIRECTORY)/$(SAMPLE).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TRAIN_DATASET)" && \
	python prepare_data.py "$(DATA_DIRECTORY)/$(TEST).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)"



# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
run_en: prepare_data
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TRAIN_DATASET)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)"


plots:
	python ../../shared_scripts/create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)" --predicted "$(OUTPUT_DIRECTORY)/$(MARKER)_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)"


run-en-exa:
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TRAIN_DATASET)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_TEST_DATASET)"

