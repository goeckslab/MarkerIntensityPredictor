import sys, os
from pathlib import Path
import mlflow
import numpy as np
from typing import List
import pandas as pd
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from library import DataLoader, FolderManagement, ExperimentHandler, RunHandler, Replacer, Reporter, Preprocessing, \
    KNNImputation
from sklearn.metrics import nan_euclidean_distances, r2_score

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
    parser.add_argument("--distance", "-d", nargs='+', action="store", help="The max distance from a centroid.")
    parser.add_argument("--phenotypes", "-ph", action="store", required=True, help="The phenotype association")
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")

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

    if len(args.distance) < 2 or len(args.distance) > 2:
        raise ValueError("Please specify only a maximum and minimum distance")

    experiment_handler: ExperimentHandler = ExperimentHandler(tracking_url=args.tracking_url)
    run_handler: RunHandler = RunHandler(tracking_url=args.tracking_url)

    FolderManagement.create_directory(path=results_folder)

    try:
        min_distance: float = float(args.distance[0])
        max_distance: float = float(args.distance[1])

        distance_grid = np.linspace(min_distance, max_distance, 10)
        print(distance_grid)

        associated_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=args.experiment)

        run_options: List = ["spatial", "no_spatial"]

        run_name: str = f"KNN Distance Based Data Imputation Percentage {args.percentage}"

        # Delete previous run
        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        # Load phenotypes
        phenotypes_per_cell: pd.DataFrame = DataLoader.load_file(load_path=args.phenotypes)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:

            # The run data
            r2_score_data: List = []
            for run_option in run_options:
                use_spatial_information: bool = True if run_option == "spatial" else False

                # Load the train dataset
                train_data, features = DataLoader.load_single_cell_data(file_name=args.file,
                                                                        keep_spatial=use_spatial_information)

                # Normalize train data set
                train_data: pd.DataFrame = Preprocessing.normalize(data=train_data.copy(), columns=features,
                                                                   create_dataframe=True)

                # Load the test dataset
                test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                                 keep_spatial=use_spatial_information)

                # Normalize test data set
                test_data: pd.DataFrame = Preprocessing.normalize(data=test_cells.copy(), columns=features,
                                                                  create_dataframe=True)

                # Replace values and return indices
                print("Replacing values...")
                replaced_test_data_cells, index_replacements = Replacer.replace_values_by_cell(data=test_data,
                                                                                               features=features,
                                                                                               percentage=args.percentage)

                # Calculate distance matrix
                print("Calculating distance matrix...")
                euclidean_distances = pd.DataFrame(data=nan_euclidean_distances(replaced_test_data_cells,
                                                                                replaced_test_data_cells,
                                                                                missing_values=0))  # distance between rows of X

                # Train knn imputer

                for distance in distance_grid:
                    print(f"Finding nearest neighbors for distance {distance}")

                    distance_mask = euclidean_distances[euclidean_distances <= distance].notnull()

                    print("Retrieving indexes...")
                    nearest_cell_indexes: pd.DataFrame = distance_mask.apply(get_indexes, axis=1)

                    if nearest_cell_indexes.apply(lambda x: len(x) == 0).any():
                        print(f"Found empty cells for distance {distance}. Skipping...")
                        continue

                    # Save for later processing
                    # Reporter.upload_csv(data=cell_cluster_information,
                    #                    file_name=f"{run_option}_{distance}_cell_cluster_information",
                    #                    save_path=results_folder)

                    print("Started imputation...")
                    for origin_cell_data in nearest_cell_indexes.iteritems():
                        imputed_data: pd.DataFrame = replaced_test_data_cells.copy()
                        origin_cell = origin_cell_data[0]
                        neighbor_cell_indexes = origin_cell_data[1]

                        # Origin cell is a first position
                        neighbor_cell_indexes.insert(0, origin_cell)

                        # Select the cell environment for the specific distance
                        cell_micro_environment: pd.DataFrame = imputed_data.iloc[neighbor_cell_indexes]

                        # replaced features for the particular cell
                        replaced_features: list = index_replacements[origin_cell]

                        # Convert 0 to nan for easier calculations
                        prepared_cell_micro_environment = cell_micro_environment.replace(0, np.nan)
                        # Select only cells which are not origin cell
                        prepared_cell_micro_environment = prepared_cell_micro_environment[1:]

                        for replaced_feature in replaced_features:
                            imputed_data.at[origin_cell, replaced_feature] = prepared_cell_micro_environment[
                                replaced_feature].mean()

                    for feature in features:
                        if "X_centroid" in feature or "Y_centroid" in feature:
                            continue

                        # Store all cell indexes, to be able to select the correct cells later for r2 comparison
                        cell_indexes_to_compare: list = []
                        for key, replaced_features in index_replacements.items():
                            if feature in replaced_features:
                                cell_indexes_to_compare.append(key)

                        r2_score_data.append({
                            "Marker": feature,
                            "Score": r2_score(test_data[feature].iloc[cell_indexes_to_compare],
                                              imputed_data[feature].iloc[cell_indexes_to_compare]),
                            "Distance": distance,
                            "Spatial": "Y" if run_option == "spatial" else "N"
                        })

            # Create dataframe
            imputed_r2_scores: pd.DataFrame = pd.DataFrame().from_records(r2_score_data)
            # Upload dataframe
            Reporter.report_r2_scores(r2_scores=imputed_r2_scores, prefix="imputed", save_path=results_folder)








































    except:
        raise

    finally:
        FolderManagement.delete_directory(path=results_folder)
