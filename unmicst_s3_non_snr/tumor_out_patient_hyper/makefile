TEST_ID=$(test_id)
MARKER=$(marker)
OUTPUT_DIRECTORY = $(TEST_ID)/$(MARKER)
DATA_DIRECTORY = ../../data/tumor_s3_non_snr
DATA_DIRECTORY_COMBINED = ../../data/tumor_s3_non_snr/combined
DATA_DIRECTORY_COMBINED_PREPROCESSED = ../../data/tumor_s3_non_snr/combined/preprocessed
DATA_DIRECTORY_PREPROCESSED = ../../data/tumor_s3_non_snr/preprocessed
PREPROCESSED_DATASET = $(TEST_ID)_preprocessed_dataset.tsv
COMBINED_DATASET = $(TEST_ID)_excluded_dataset.csv
COMBINED_PREPROCESSED_DATA = $(TEST_ID)_excluded_dataset.tsv

# Create preprocessed dataset for combined data
"$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_DATASET)":
	mkdir -p "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)" && python prepare_data.py "$(DATA_DIRECTORY_COMBINED)/$(COMBINED_DATASET)" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)"

# Create preprocessed dataset for single data
"$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)":
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && python prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"


# Create combined train set
"$(DATA_DIRECTORY_COMBINED)/combined":
	mkdir -p "$(DATA_DIRECTORY_COMBINED)" && python combine_datasets.py --dir "$(DATA_DIRECTORY)" --target "$(TEST_ID)" --output_dir "$(DATA_DIRECTORY_COMBINED)"

# Create final configuration file.
"$(TEST_ID)_$(MARKER)_final_config.yaml": "$(DATA_DIRECTORY_COMBINED)/combined"
	python create_input_features.py "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(TEST_ID)_$(MARKER)_input_features.yaml";
	python create_experiment_config.py "$(TEST_ID)" "$(MARKER)" "base_config.yaml";
	cat "$(TEST_ID)_$(MARKER)_input_features.yaml" base_config.yaml > "$(TEST_ID)_$(MARKER)_final_config.yaml"

# Create the hyperopt file.
"$(TEST_ID)_$(MARKER)_hyperopt_config.yaml": "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_DATASET)"
	python create_input_features.py "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(TEST_ID)_$(MARKER)_input_features.yaml"
	python create_hyperopt.py "$(MARKER)" "$(TEST_ID)_$(MARKER)_hyperopt.yaml"; cat "$(TEST_ID)_$(MARKER)_input_features.yaml" "$(TEST_ID)_$(MARKER)_hyperopt.yaml" > "$(TEST_ID)_$(MARKER)_hyperopt_config.yaml"

ludwig-hyperopt: "$(DATA_DIRECTORY_COMBINED)/combined" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(TEST_ID)_$(MARKER)_hyperopt_config.yaml"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig hyperopt --dataset "../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)" --config "../../$(TEST_ID)_$(MARKER)_hyperopt_config.yaml"

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(TEST_ID) directory.
ludwig-experiment: "$(TEST_ID)_$(MARKER)_final_config.yaml"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig experiment --dataset "../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)" --config "../../$(TEST_ID)_$(MARKER)_final_config.yaml" -rs 456

ludwig-evaluate:
	python prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig evaluate --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" --model_path ./results/experiment_run/model --output_directory evaluate/$(TEST_ID)

ludwig-plots:
	python create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --predicted "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/${MARKER}_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/"

clean:
	rm "$(TEST_ID)_$(MARKER)_hyperopt.yaml" "$(TEST_ID)_$(MARKER)_hyperopt_config.yaml"  "$(TEST_ID)_$(MARKER)_final_config.yaml" "$(TEST_ID)_$(MARKER)_input_features.yaml" "base_config.yaml";
