# Compares the cell distances used by the KNN imputer between included spatial data or not
import mlflow
from pathlib import Path
import time
from library.mlflow_helper.experiment_handler import ExperimentHandler
import argparse
from library.data.data_loader import DataLoader
import pandas as pd
from library.preprocessing.preprocessing import Preprocessing
from library.preprocessing.replacements import Replacer
from library.data.folder_management import FolderManagement
from library.mlflow_helper.reporter import Reporter
from library.plotting.plots import Plotting
from sklearn.metrics.pairwise import nan_euclidean_distances, euclidean_distances
import numpy as np
from matplotlib.cbook import boxplot_stats
from typing import Dict

base_path = "knn_imputation_comparison"


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
    parser.add_argument("--file", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training the KNN")
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Started knn imputation...")
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "KNN Distance Comparison"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                                create_experiment=True)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found!")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(base_path)

    try:

        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=f"{run_name} Percentage {args.percentage}") as run:
            plotter: Plotting = Plotting(base_path=base_path, args=args)

            # The distances for each run. Spatial and No Spatial
            euclidean_distances_all_cells: Dict = {}
            micron_distances_all_cells: Dict = {}
            run_distances_per_neighbor: Dict = {}

            run_options: list = ["No Spatial", "Spatial"]

            for run_option in run_options:
                print(f"Processing {run_option}")
                use_spatial_information: bool = True if run_option == "Spatial" else False

                with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                                      run_name=f"{run_option}") as ml_run_option:
                    mlflow.log_param("Percentage of replaced values", args.percentage)
                    mlflow.log_param("Files", args.file)
                    mlflow.log_param("Seed", args.seed)
                    mlflow.set_tag("Percentage", args.percentage)
                    mlflow.log_param("Keep morph", args.morph)
                    mlflow.log_param("Keep spatial", True if run_option == "Spatial" else False)

                    train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                        file_to_exclude=args.file,
                                                                                        keep_spatial=use_spatial_information)
                    train_data = pd.DataFrame(data=Preprocessing.normalize(train_cells.copy()), columns=features)

                    test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                                     keep_spatial=use_spatial_information)
                    test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

                    replaced_test_data_cells, index_replacements = Replacer.replace_values_by_cell(data=test_data,
                                                                                                   features=features,
                                                                                                   percentage=args.percentage)

                    distances = pd.DataFrame(data=nan_euclidean_distances(replaced_test_data_cells,
                                                                          replaced_test_data_cells,
                                                                          missing_values=0))  # distance between rows of X

                    # Get the indices of the nearest neighbors
                    N = 3
                    nearest_neighbors_indices = pd.DataFrame(
                        distances.index[np.argsort(-distances.values, axis=0)[-1:-1 - N:-1]], columns=distances.columns)

                    euclidean_distances_per_cell: pd.DataFrame = pd.DataFrame()
                    micron_distances_per_cell: pd.DataFrame = pd.DataFrame()
                    distances_per_neighbor = pd.DataFrame()

                    # Load dataframe again to include x and y data
                    if run_option == "No Spatial":
                        test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                                         keep_spatial=True)

                    for cellId, nearestNeighbors in nearest_neighbors_indices.iteritems():
                        cell = test_cells.iloc[[cellId]][["X_centroid", "Y_centroid"]].reset_index(drop=True)
                        first_neighbor = test_cells.iloc[[nearestNeighbors.iloc[1]]][
                            ["X_centroid", "Y_centroid"]].reset_index(drop=True)
                        second_neighbor = test_cells.iloc[[nearestNeighbors.iloc[2]]][
                            ["X_centroid", "Y_centroid"]].reset_index(drop=True)

                        frames = [cell, first_neighbor, second_neighbor]
                        new_df = pd.concat(frames)
                        euclidian_distances = euclidean_distances(new_df)

                        euclidean_distances_per_cell = euclidean_distances_per_cell.append({
                            "Cell": cellId,
                            "First Neighbor": euclidian_distances[0][1],
                            "Second Neighbor": euclidian_distances[0][2],
                        }, ignore_index=True)

                        distances_per_neighbor = distances_per_neighbor.append({
                            "Neighbor": "First",
                            "Distance": euclidian_distances[0][1]
                        }, ignore_index=True)

                        distances_per_neighbor = distances_per_neighbor.append({
                            "Neighbor": "Second",
                            "Distance": euclidian_distances[0][2]
                        }, ignore_index=True)

                    plotter.scatter_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                                         title="Spatial Distances", file_name="Spatial Distances")

                    euclidean_distances_all_cells[run_option] = euclidean_distances_per_cell
                    run_distances_per_neighbor[run_option] = distances_per_neighbor

            for key in euclidean_distances_all_cells.keys():
                prefix = "no_spatial" if key == "No Spatial" else "spatial"
                Reporter.upload_csv(data=euclidean_distances_all_cells[key], file_name=f"{prefix}_euclidean_distances",
                                    save_path=base_path)

            for key in run_distances_per_neighbor.keys():
                prefix = "no_spatial" if key == "No Spatial" else "spatial"
                Reporter.upload_csv(data=run_distances_per_neighbor[key], file_name=f"{prefix}_distance_neighbors",
                                    save_path=base_path)

            outlier_count: pd.DataFrame = pd.DataFrame()
            for key, distances in run_distances_per_neighbor.items():
                outlier_count = outlier_count.append({
                    "Combination": f"{key}_first",
                    "Count": len(
                        [y for stat in boxplot_stats(distances[["Neighbor"] == 'First']['Distance']) for y in
                         stat['fliers']])
                }, ignore_index=True)

                outlier_count = outlier_count.append({
                    "Combination": f"{key}_second",
                    "Count": len(
                        [y for stat in boxplot_stats(distances[["Neighbor"] == 'Second']['Distance']) for y in
                         stat['fliers']])
                }, ignore_index=True)

            Reporter.upload_csv(data=outlier_count, save_path=base_path, file_name="outlier_count")
            plotter.box_plot(data=run_distances_per_neighbor, x="Neighbor", y="Distance",
                             title="Neighbor Spatial Distances", file_name="Neighbor Spatial Distances")



    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
