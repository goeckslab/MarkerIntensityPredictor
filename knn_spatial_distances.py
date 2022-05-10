# Checks whether the nearest neighbors used by the KNN imputer are also the spatial nearest neighbors
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
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np

base_path = "knn_spatial_distances"


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be evaluated",
                        default="Default", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--file", action="store", required=True, help="The file to use for imputation")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training KNN and simple imputer")
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Started knn spatial neighbor detection...")
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "KNN Distance Comparison"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found!")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(base_path)

    try:

        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=f"{run_name} Percentage {args.percentage}") as run:

            # The distances for each run. Spatial and No Spatial
            run_distances: dict = {}

            mlflow.log_param("Percentage of replaced values", args.percentage)
            mlflow.log_param("Files", args.file)
            mlflow.log_param("Seed", args.seed)
            mlflow.set_tag("Percentage", args.percentage)
            mlflow.log_param("Keep morph", args.morph)

            train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                file_to_exclude=args.file,
                                                                                keep_spatial=True)
            train_data = pd.DataFrame(data=Preprocessing.normalize(train_cells.copy()), columns=features)

            test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                             keep_spatial=True)
            test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

            replaced_test_data_cells, index_replacements = Replacer.replace_values_by_cell(data=test_data,
                                                                                           features=features,
                                                                                           percentage=args.percentage)

            distances = pd.DataFrame(
                data=nan_euclidean_distances(replaced_test_data_cells,
                                             replaced_test_data_cells,
                                             missing_values=0))  # distance between rows of X

            # Get the indices of the nearest neighbors
            N = 3
            nearest_neighbors_indices = pd.DataFrame(
                distances.index[np.argsort(-distances.values, axis=0)[-1:-1 - N:-1]], columns=distances.columns)
            print(nearest_neighbors_indices)

            spatial_information = pd.DataFrame()

            for cellId, nearestNeighbors in nearest_neighbors_indices.iteritems():
                cell = test_data.iloc[[cellId]][["X_centroid", "Y_centroid", "Area"]].reset_index(drop=True)
                first_neighbor = test_data.iloc[[nearestNeighbors.iloc[1]]][
                    ["X_centroid", "Y_centroid", "Area"]].reset_index(drop=True)
                second_neighbor = test_data.iloc[[nearestNeighbors.iloc[2]]][
                    ["X_centroid", "Y_centroid", "Area"]].reset_index(drop=True)

                first_far_x = "F" if abs(
                    cell.iloc[0]["X_centroid"] - first_neighbor.iloc[0]["X_centroid"]) >= 500 else "N"
                first_far_y = "F" if abs(
                    cell.iloc[0]["Y_centroid"] - first_neighbor.iloc[0]["Y_centroid"]) >= 500 else "N"

                second_far_x = "F" if abs(
                    cell.iloc[0]["X_centroid"] - second_neighbor.iloc[0]["X_centroid"]) >= 500 else "N"
                second_far_y = "F" if abs(
                    cell.iloc[0]["Y_centroid"] - second_neighbor.iloc[0]["Y_centroid"]) >= 500 else "N"

                spatial_information = spatial_information.append({
                    "first_neighbor_x": first_far_x,
                    "first_neighbor_y": first_far_y,
                    "second_neighbor_x": second_far_x,
                    "second_neighbor_y": second_far_y,
                }, ignore_index=True)

            print(spatial_information)
            input()
            lowest_distances = pd.DataFrame()
            lowest_distances['first_neighbor'], lowest_distances['second_neighbor'] = np.sort(distances,
                                                                                              axis=1)[:, 1:3].T

            # scaler = MinMaxScaler()
            # lowest_distances = pd.DataFrame(data=scaler.fit_transform(lowest_distances),
            #                                columns=['first_neighbor', 'second_neighbor'])



    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
