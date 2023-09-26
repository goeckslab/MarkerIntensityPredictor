DATA_DIRECTORY = ../../data/tumor_mesmer
DATA_DIRECTORY_PREPROCESSED = ../../data/tumor_mesmer/preprocessed
SAMPLE=$(sample_id)
PREPROCESSED_DATASET = $(SAMPLE)_preprocessed_dataset.tsv
TEST_ID=$(test_id)
MARKER=$(marker)
OUTPUT_DIRECTORY = $(SAMPLE)/$(MARKER)

# Create preprocessed dataset.
"$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)":
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && python ../../shared_scripts/prepare_data.py "$(DATA_DIRECTORY)/$(SAMPLE).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"

# Create final configuration file.
"$(SAMPLE)_final_config.yaml":
	python ../../shared_scripts/create_input_features.py "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(SAMPLE)_$(MARKER)_input_features.yaml";
	python ../../shared_scripts/create_experiment_config.py "$(SAMPLE)" "$(MARKER)" "base_config.yaml";
	cat "$(SAMPLE)_$(MARKER)_input_features.yaml" base_config.yaml > "$(SAMPLE)_$(MARKER)_final_config.yaml"

# Create the hyperopt file.
"$(SAMPLE)_hyperopt_config.yaml": 
	python ../../shared_scripts/create_input_features.py "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" "$(MARKER)" "$(SAMPLE)_$(MARKER)_input_features.yaml"
	python ../../shared_scripts/create_hyperopt.py "$(MARKER)" "$(SAMPLE)_$(MARKER)_hyperopt.yaml"; cat "$(SAMPLE)_$(MARKER)_input_features.yaml" "$(SAMPLE)_$(MARKER)_hyperopt.yaml" > "$(SAMPLE)_$(MARKER)_hyperopt_config.yaml"


ludwig-hyperopt: "$(SAMPLE)_hyperopt_config.yaml" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig hyperopt --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" --config "../../$(SAMPLE)_$(MARKER)_hyperopt_config.yaml"

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
ludwig-experiment: "$(SAMPLE)_final_config.yaml" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig experiment --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" --config "../../$(SAMPLE)_$(MARKER)_final_config.yaml" -rs 456

ludwig-evaluate:
	python ../../shared_scripts/prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv"
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && ludwig evaluate --dataset "../../$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --model_path ./results/experiment_run/model --output_directory evaluate/$(TEST_ID)

ludwig-plots:
	python ../../shared_scripts/create_plots.py --truth "$(DATA_DIRECTORY_PREPROCESSED)/$(TEST_ID)_preprocessed_dataset.tsv" --predicted "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/${MARKER}_predictions.csv" --marker "${MARKER}" --output "$(OUTPUT_DIRECTORY)/evaluate/$(TEST_ID)/"

clean:
	rm -rf "$(SAMPLE)/$(MARKER)/results/hyperopt"
	rm "$(SAMPLE)_$(MARKER)_hyperopt.yaml" "$(SAMPLE)_$(MARKER)_hyperopt_config.yaml"  "$(SAMPLE)_$(MARKER)_final_config.yaml" "$(SAMPLE)_$(MARKER)_input_features.yaml" "base_config.yaml";
