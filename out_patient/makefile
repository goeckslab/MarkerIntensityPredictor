TEST_ID=$(test_id)
MARKER=$(marker)
OUTPUT_DIRECTORY = $(TEST_ID)/$(MARKER)
DATA_DIRECTORY = ../data/Tumor
DATA_DIRECTORY_COMBINED = ../data/Tumor/combined
DATA_DIRECTORY_COMBINED_PREPROCESSED = ../data/Tumor/combined/preprocessed
DATA_DIRECTORY_PREPROCESSED = ../data/Tumor/preprocessed
PREPROCESSED_DATASET = $(TEST_ID)_excluded_dataset.tsv
COMBINED_DATASET = $(TEST_ID)_excluded_dataset.csv



# Create preprocessed dataset for combined data
"$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)":
	mkdir -p "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)" && python prepare_data.py "$(DATA_DIRECTORY_COMBINED)/$(COMBINED_DATASET)" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)"

# Create preprocessed dataset for single data
"$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv":
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && python prepare_data.py "$(DATA_DIRECTORY)/$(SAMPLE).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"

# Create combined train set
"$(DATA_DIRECTORY_COMBINED)/combined":
	mkdir -p "$(DATA_DIRECTORY_COMBINED)" && python combine_datasets.py --dir "$(DATA_DIRECTORY)" --target "$(TEST_ID)" --output_dir "$(DATA_DIRECTORY_COMBINED)"

# Create final configuration file.
"$(TEST_ID)_final_config.yaml": create_input_features.py base_config.yaml "$(DATA_DIRECTORY_COMBINED)/combined" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)"
	python create_input_features.py "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(TEST_ID)_input_features.yaml"; cat "$(TEST_ID)_input_features.yaml" base_config.yaml > "$(TEST_ID)_final_config.yaml"


# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(TEST_ID) directory.
ludwig-experiment: "$(TEST_ID)_final_config.yaml"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig experiment --dataset "../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)" --config "../../$(TEST_ID)_final_config.yaml" -rs 456

ludwig-evaluate:
	python prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig evaluate --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --model_path ./results/experiment_run/model --output_directory evaluate/$(TEST_ID)

ludwig-plots:
	python create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --predicted "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/${MARKER}_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/"

clean:
	rm "$(TEST_ID)_final_config.yaml" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(TEST_ID)_input_features.yaml"

