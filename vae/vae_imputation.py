import os, sys
import pandas as pd
import argparse
from pathlib import Path
import time
import mlflow
from typing import Dict, List
from mlflow.entities import Run

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import ExperimentHandler, RunHandler, DataLoader, FolderManagement, Replacer, SplitHandler, Reporter, \
    FeatureEngineer, Preprocessing
from library.vae.vae import MarkerPredictionVAE
from library.vae.vae_imputer import VAEImputer

results_folder = Path("vae_imputation")


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be used to store the results",
                        default="Default", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--iterations", "-i", action="store", help="The iterations used for imputation",
                        default=2, required=True, type=int)
    parser.add_argument("--phenotypes", "-ph", action="store", required=True, help="The phenotype association")
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training the VAE")
    parser.add_argument("--index_replacements", "-ir", action="store", required=False,
                        help="The index replacements with markers to replicate experiments")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    results_folder = Path(f"{results_folder}_{str(int(time.time_ns() / 1000))}")

    experiment_handler: ExperimentHandler = ExperimentHandler(tracking_url=args.tracking_url)
    run_handler: RunHandler = RunHandler(tracking_url=args.tracking_url)

    experiment_name = args.experiment
    # The id of the associated
    associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                            create_experiment=True)

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(path=results_folder)

    imputed_cell_data: List = []
    replaced_cells_data: List = []
    test_cell_data: List = []
    train_cell_data: List = []
    val_cell_data: List = []
    un_normalized_test_cell_data: List = []

    try:
        run_name: str = f"VAE Data Imputation Percentage {args.percentage}"

        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name):

            radius_grid = [10, 20, 30, 50, 75, 100]

            cells, marker_columns = DataLoader.load_single_cell_data(file_name=args.file, keep_spatial=True,
                                                                     return_df=True, return_marker_columns=True)
            features_to_impute = list(cells.columns)

            if "X_centroid" in features_to_impute:
                features_to_impute.remove("X_centroid")

            if "Y_centroid" in features_to_impute:
                features_to_impute.remove("Y_centroid")

            Reporter.upload_csv(data=pd.Series(features_to_impute), file_name="features_to_impute",
                                save_path=results_folder)

            if args.index_replacements is None:
                # Create replacements
                index_replacements: Dict = Replacer.select_index_and_features_to_replace(features=list(cells.columns),
                                                                                         length_of_data=cells.shape[0],
                                                                                         percentage=args.percentage)
            else:
                index_replacements_df: pd.DataFrame = pd.read_csv(args.index_replacements)
                values: List = index_replacements_df.values.tolist()
                index_replacements: Dict = {}
                # Convert dataframe back to expected dictionary
                for i, value in enumerate(values):
                    index_replacements[i] = value

            Reporter.upload_csv(data=pd.DataFrame.from_dict(index_replacements).T, file_name="index_replacements",
                                save_path=results_folder)

            bulk_engineer: FeatureEngineer = FeatureEngineer(radius=10, folder_name=args.folder,
                                                             file_to_exclude=args.file)
            test_data_engineer: FeatureEngineer = FeatureEngineer(radius=10, file_path=args.file)

            for radius in radius_grid:
                print(f"Processing radius {radius}")
                folder_name: str = f"radius_{radius}"
                bulk_engineer.radius = radius

                bulk_engineer.create_features()

                mlflow.log_param("Marker Count", len(bulk_engineer.marker_columns))

                # Report to mlflow
                for key, data in bulk_engineer.feature_engineered_data.items():
                    Reporter.upload_csv(data=data, save_path=results_folder, file_name=f"engineered_{key}",
                                        mlflow_folder=folder_name)

                train_data = pd.concat(list(bulk_engineer.feature_engineered_data.values()))

                train_data, val_data = SplitHandler.create_splits(cells=train_data,
                                                                  features=list(train_data.columns),
                                                                  create_val=False)

                train_data, std_scaler = Preprocessing.normalize_feature_engineered_data(data=train_data)
                val_data, _, = Preprocessing.normalize_feature_engineered_data(data=val_data,
                                                                               standard_scaler=std_scaler)

                test_data_engineer.radius = radius
                test_data_engineer.create_features()

                test_data = list(test_data_engineer.feature_engineered_data.values())[0]

                temp_data = test_data.copy()
                temp_data["Radius"] = radius
                un_normalized_test_cell_data.append(temp_data)

                test_data, _ = Preprocessing.normalize_feature_engineered_data(data=test_data,
                                                                               standard_scaler=std_scaler)

                # Report to mlflow
                Reporter.upload_csv(data=test_data, save_path=results_folder, file_name=f"test_data_engineered",
                                    mlflow_folder=folder_name)

                replaced_test_data = Replacer.replace_values_by_cell(data=test_data,
                                                                     index_replacements=index_replacements,
                                                                     value_to_replace=0)

                # Report to mlflow
                Reporter.upload_csv(data=replaced_test_data, file_name="replaced_test_data",
                                    mlflow_folder=folder_name,
                                    save_path=results_folder)

                columns_to_select = list(set(replaced_test_data.columns) - {"X_centroid", "Y_centroid", "Phenotype",
                                                                            "Cell Neighborhood"})
                print("Imputing data")
                vae, encoder, decoder, history = MarkerPredictionVAE.build_5_layer_variational_auto_encoder(
                    training_data=train_data[columns_to_select],
                    validation_data=val_data[columns_to_select],
                    input_dimensions=train_data[columns_to_select].shape[1],
                    embedding_dimension=10)

                vae_imputer: VAEImputer = VAEImputer(model=vae, index_replacements=index_replacements,
                                                     replaced_data=replaced_test_data[columns_to_select],
                                                     iterations=args.iterations,
                                                     features_to_impute=features_to_impute)
                vae_imputer.impute()

                imputed_cells = vae_imputer.imputed_data
                Reporter.upload_csv(data=imputed_cells, file_name="imputed_cells", mlflow_folder=folder_name,
                                    save_path=results_folder)
                Reporter.upload_csv(data=replaced_test_data, file_name="replaced_test_data",
                                    mlflow_folder=folder_name,
                                    save_path=results_folder)

                imputed_cells["Radius"] = radius
                imputed_cell_data.append(imputed_cells)

                replaced_test_data["Radius"] = radius
                replaced_cells_data.append(replaced_test_data)

                test_data["Radius"] = radius
                test_cell_data.append(test_data)

                val_data["Radius"] = radius
                val_cell_data.append(val_data)

                train_data["Radius"] = radius
                train_cell_data.append(train_data)

            combined_imputed_cells = pd.concat([data for data in imputed_cell_data])
            combined_replaced_cells = pd.concat([data for data in replaced_cells_data])
            combined_test_cells = pd.concat([data for data in test_cell_data])
            combined_un_normalized_cells = pd.concat([data for data in un_normalized_test_cell_data])
            combined_train_cells = pd.concat([data for data in train_cell_data])
            combined_val_cells = pd.concat([data for data in val_cell_data])

            Reporter.upload_csv(data=combined_imputed_cells, file_name="combined_imputed_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_replaced_cells, file_name="combined_replaced_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_test_cells, file_name="combined_test_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_un_normalized_cells, file_name="combined_un_normalized_test_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_train_cells, file_name="combined_train_cells",
                                save_path=results_folder)
            Reporter.upload_csv(data=combined_val_cells, file_name="combined_val_cells",
                                save_path=results_folder)

    except:
        raise
    finally:
        FolderManagement.delete_directory(path=results_folder)
