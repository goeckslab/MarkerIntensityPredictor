DATA_DIRECTORY = ../data
SAMPLE = 9_2_1
PREPROCESSED_DATASET = $(SAMPLE)_preprocessed_dataset.tsv
TEST_ID = 9_15_2
MARKER = CK19
OUTPUT_DIRECTORY = $(SAMPLE)-$(MARKER)

# Create preprocessed dataset.
$(PREPROCESSED_DATASET):
	python prepare_data.py "$(DATA_DIRECTORY)/Tumor/$(SAMPLE).csv" $(PREPROCESSED_DATASET)

# Create final configuration file.
$(SAMPLE)_final_config.yaml: create_input_features.py base_config.yaml $(PREPROCESSED_DATASET)
	python create_input_features.py $(PREPROCESSED_DATASET) $(MARKER) $(SAMPLE)_input_features.yaml; cat $(SAMPLE)_input_features.yaml base_config.yaml > $(SAMPLE)_final_config.yaml

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
ludwig-experiment: $(SAMPLE)_final_config.yaml $(PREPROCESSED_DATASET)
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig experiment --dataset ../$(PREPROCESSED_DATASET) --config ../$(SAMPLE)_final_config.yaml -rs 456

ludwig-evaluate:
	python prepare_data.py "$(DATA_DIRECTORY)/Tumor/$(TEST_ID).csv" $(TEST_ID)_preprocessed_dataset.tsv
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig evaluate --dataset ../$(TEST_ID)_preprocessed_dataset.tsv --model_path ./results/experiment_run/model --output_directory evaluate/$(TEST_ID)

clean:
	rm $(SAMPLE)_final_config.yaml $(PREPROCESSED_DATASET)

