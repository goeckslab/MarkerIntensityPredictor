import sys, os
from pathlib import Path
import mlflow
import numpy as np
from typing import List, Dict
import pandas as pd
import time
from sklearn.impute import KNNImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from library import DataLoader, FolderManagement, ExperimentHandler, RunHandler, Replacer, Reporter, KNNImputation, \
    FeatureEngineer, Preprocessing

results_folder = Path("knn_distance_imputation")


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
    parser.add_argument("--phenotypes", "-ph", action="store", required=True, help="The phenotype association")
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training the KNN")
    parser.add_argument("--index_replacements", "-ir", action="store", required=False,
                        help="The index replacements with markers to replicate experiments")
    parser.add_argument("--preprocessing", "-pre", action="store", required=False, choices=["n", "s"], default="s",
                        help="The preprocessing to normalize/standardize the data", type=str)

    return parser.parse_args()


def get_indexes(x):
    # Drop self nearest index
    x.drop(labels=[x.name], inplace=True)
    return list(x[x].index.values)


def map_indexes_to_phenotype(x, phenotypes: pd.DataFrame):
    phenotypes = [phenotypes.iloc[index].values[0] for index in x]
    return phenotypes


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

    try:
        n_neighbors: int = 6
        radius_grid = [10, 20, 30, 50, 75, 100]

        run_name: str = f"KNN Distance Based Imputation Percentage {args.percentage} S" \
            if args.preprocessing == "s" \
            else f"KNN Distance Based Imputation Percentage {args.percentage} N"

        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        cells, marker_columns = DataLoader.load_single_cell_data(file_name=args.file, keep_spatial=True, return_df=True,
                                                                 return_marker_columns=True)

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

        imputed_cell_data: List = []
        replaced_cells_data: List = []
        test_cell_data: List = []
        train_cell_data: List = []
        un_normalized_test_cell_data: List = []

        bulk_engineer: FeatureEngineer = FeatureEngineer(folder_name=args.folder, file_to_exclude=args.file,
                                                         radius=0)

        test_data_engineer = FeatureEngineer = FeatureEngineer(file_path=args.file, radius=0)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:
            mlflow.log_param("n Neighbors", n_neighbors)
            mlflow.log_param("Preprocessing Method", "Standardize" if args.preprocessing == "s" else "Normalize")
            # Upload index replacement for data analysis
            Reporter.upload_csv(data=pd.DataFrame.from_dict(index_replacements).T, file_name="index_replacements",
                                save_path=results_folder)

            for radius in radius_grid:
                folder_name: str = f"radius_{radius}"

                bulk_engineer.radius = radius
                print(f"Radius {bulk_engineer.radius}")
                bulk_engineer.create_features()

                mlflow.log_param("Marker Count", len(bulk_engineer.marker_columns))

                # Report to mlflow
                for key, data in bulk_engineer.feature_engineered_data.items():
                    Reporter.upload_csv(data=data, save_path=results_folder, file_name=f"engineered_{key}",
                                        mlflow_folder=folder_name)

                train_data = pd.concat(list(bulk_engineer.feature_engineered_data.values()))

                if args.preprocessing == "s":
                    train_data, scaler = Preprocessing.standardize_feature_engineered_data(data=train_data)
                else:
                    train_data, scaler = Preprocessing.normalize_feature_engineered_data(data=train_data)

                test_data_engineer.radius = radius
                test_data_engineer.create_features()

                test_data = list(test_data_engineer.feature_engineered_data.values())[0]

                temp_data = test_data.copy()
                temp_data["Radius"] = radius
                un_normalized_test_cell_data.append(temp_data)

                if args.preprocessing == "s":
                    test_data, _ = Preprocessing.standardize_feature_engineered_data(data=test_data,
                                                                                     scaler=scaler)
                else:
                    test_data, _ = Preprocessing.normalize_feature_engineered_data(data=test_data,
                                                                                   scaler=scaler)

                # Report to mlflow
                Reporter.upload_csv(data=test_data, save_path=results_folder, file_name=f"test_data_engineered",
                                    mlflow_folder=folder_name)

                replaced_test_data = Replacer.replace_values_by_cell(data=test_data,
                                                                     index_replacements=index_replacements,
                                                                     value_to_replace=np.nan)

                # Report to mlflow
                Reporter.upload_csv(data=replaced_test_data, file_name="replaced_test_data", mlflow_folder=folder_name,
                                    save_path=results_folder)

                columns_to_select = list(set(replaced_test_data.columns) - {"X_centroid", "Y_centroid", "Phenotype",
                                                                            "Cell Neighborhood"})
                print("Imputing data")
                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed_cells = KNNImputation.impute(train_data=train_data[columns_to_select],
                                                     test_data=replaced_test_data[columns_to_select],
                                                     missing_values=np.nan)

                Reporter.upload_csv(data=imputed_cells, file_name="imputed_cells", mlflow_folder=folder_name,
                                    save_path=results_folder)

                imputed_cells["Radius"] = radius
                imputed_cell_data.append(imputed_cells)

                replaced_test_data["Radius"] = radius
                replaced_cells_data.append(replaced_test_data)

                test_data["Radius"] = radius
                test_cell_data.append(test_data)

                train_data["Radius"] = radius
                train_cell_data.append(train_data)

            combined_imputed_cells = pd.concat([data for data in imputed_cell_data])
            combined_replaced_cells = pd.concat([data for data in replaced_cells_data])
            combined_test_cells = pd.concat([data for data in test_cell_data])
            combined_un_normalized_test_cells = pd.concat([data for data in un_normalized_test_cell_data])
            combined_train_cells = pd.concat([data for data in train_cell_data])

            Reporter.upload_csv(data=combined_imputed_cells, file_name="combined_imputed_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_replaced_cells, file_name="combined_replaced_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_test_cells, file_name="combined_test_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_train_cells, file_name="combined_train_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_un_normalized_test_cells, file_name="combined_un_normalized_test_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=test_data_engineer.phenotypes[Path(args.file).stem],
                                file_name="test_data_phenotypes",
                                save_path=results_folder)

            features_to_impute = list(cells.columns)

            if "X_centroid" in features_to_impute:
                features_to_impute.remove("X_centroid")

            if "Y_centroid" in features_to_impute:
                features_to_impute.remove("Y_centroid")

            Reporter.upload_csv(data=pd.Series(features_to_impute), file_name="features_to_impute",
                                save_path=results_folder)


    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(path=results_folder)
