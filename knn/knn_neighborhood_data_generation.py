# Compares the cell distances used by the KNN imputer between included spatial data or not

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import mlflow
from pathlib import Path
import time
import argparse
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances, euclidean_distances
import numpy as np
from matplotlib.cbook import boxplot_stats
from typing import Dict, List
from library import DataLoader, FolderManagement, PhenotypeMapper, TTest, Preprocessing, Reporter, ExperimentHandler, \
    RunHandler, Replacer, KNNImputation

base_path = "knn_neighbor_data_generation"


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
    parser.add_argument("--phenotypes", "-ph", action="store", required=False, help="The phenotype association")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Started knn data generation...")
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = f"KNN Neighborhood Data Generation Percentage {args.percentage}"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    experiment_name = args.experiment
    # The id of the associated
    associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                            create_experiment=True)

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(base_path)

    try:
        # Delete previous run
        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=run_name) as run:

            # The euclidean distances for each run. Spatial and No Spatial
            euclidean_distances_all_cells: Dict = {}
            micron_distances_all_cells: Dict = {}

            # Counts of neighbors for all cells. As in First Neighbor Immune 1000
            neighboring_phenotype_count: Dict = {}

            # The euclidean distances for each cell, just in a different setup
            run_distances_per_neighbor: Dict = {}

            mapped_phenotype_data: List = []

            run_options: list = ["no_spatial", "spatial"]
            min_neighbors = 2
            max_neighbors = 7
            amount_of_neighbors = np.arange(min_neighbors, max_neighbors)

            # Load the dataset for data information collection
            test_cells, features = DataLoader.load_single_cell_data(file_name=args.file,
                                                                    keep_spatial=True)

            # Create index replacements
            index_replacements: Dict = Replacer.select_index_and_features_to_replace(features=features,
                                                                                     length_of_data=test_cells.shape[0],
                                                                                     percentage=args.percentage)

            # Upload index replacement for data analysis
            Reporter.upload_csv(data=pd.DataFrame.from_dict(index_replacements).T, file_name="index_replacements",
                                save_path=base_path)

            # R2 data for all runs and all options
            combined_r2_score_data: List = []
            combined_nearest_neighbor_indices_data: List = []
            combined_mapped_phenotype_data: List = []
            combined_imputed_data: List = []
            combined_replaced_cell_data: List = []
            combined_euclidean_distances_data: List = []

            for run_option in run_options:
                print(f"Processing {run_option}")
                spatial: bool = True if run_option == 'spatial' else False

                train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                    file_to_exclude=args.file,
                                                                                    keep_spatial=spatial)

                mlflow.log_param("Run Option Files Used", files_used)

                train_data = pd.DataFrame(data=Preprocessing.normalize(train_cells.copy()), columns=features)

                # Load the test dataset
                test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                                 keep_spatial=spatial)

                # Normalize test data set
                test_data: pd.DataFrame = Preprocessing.normalize(data=test_cells.copy(), columns=features,
                                                                  create_dataframe=True)

                # Replace values and return indices
                replaced_test_data_cells = Replacer.replace_values_by_cell(data=test_data,
                                                                           index_replacements=index_replacements)

                distances = pd.DataFrame(data=nan_euclidean_distances(replaced_test_data_cells,
                                                                      replaced_test_data_cells,
                                                                      missing_values=0))  # distance between rows of X

                # Reporter.upload_csv(data=distances, file_name="euclidean_distances", save_path=base_path)
                Reporter.upload_csv(data=test_data, file_name="normalized_test_data", save_path=base_path)

                for neighbor_count in amount_of_neighbors:
                    print(f"Processing {neighbor_count} neighbors... ")
                    folder_name: str = f"{'spatial' if spatial else 'no_spatial'}_{neighbor_count}"
                    # Get the indices of the nearest neighbors

                    N = neighbor_count + 1
                    # Select the nearest neighbors ascending
                    nearest_neighbors_indices = pd.DataFrame(np.argsort(distances.values, axis=0)[:N],
                                                             columns=distances.columns)

                    # Upload nearest neighbor indices
                    Reporter.upload_csv(data=nearest_neighbors_indices, save_path=base_path,
                                        file_name="nearest_neighbor_indices", mlflow_folder=folder_name)

                    cell_phenotypes: pd.DataFrame = DataLoader.load_file(args.phenotypes)
                    cell_phenotypes.drop(columns=["imageid"], inplace=True)
                    phenotypes = PhenotypeMapper.map_nn_to_phenotype(nearest_neighbors=nearest_neighbors_indices,
                                                                     phenotypes=cell_phenotypes,
                                                                     neighbor_count=neighbor_count)
                    # Add new information about cell origin phenotypes
                    phenotypes = phenotypes.T
                    phenotypes["Base Cell"] = cell_phenotypes["phenotype"].values
                    Reporter.upload_csv(data=phenotypes, save_path=base_path, file_name="mapped_phenotypes",
                                        mlflow_folder=folder_name)

                    # Save data
                    euclidean_distances_per_cell_data: List = []

                    # Load dataframe again to include x and y data
                    if not spatial:
                        test_cells, features = DataLoader.load_single_cell_data(file_name=args.file,
                                                                                keep_spatial=True)
                        # Normalize test data set
                        test_data: pd.DataFrame = Preprocessing.normalize(data=test_cells.copy(), columns=features,
                                                                          create_dataframe=True)

                    print("Identifying nearest neighbors and distances...")
                    for cellId, nearestNeighbors in tqdm(nearest_neighbors_indices.iteritems()):
                        origin = test_data.iloc[[cellId]][["X_centroid", "Y_centroid"]].reset_index(drop=True)
                        frames = [origin]
                        for i, neighbors in enumerate(nearestNeighbors):
                            # Skip first cell
                            if i == 0:
                                continue
                            frames.append(test_data.iloc[[nearestNeighbors.iloc[i]]][
                                              ["X_centroid", "Y_centroid"]].reset_index(drop=True))

                        # Create cell neighborhood
                        cell_neighbor_hood: pd.DataFrame = pd.concat(frames).reset_index(drop=True)
                        euclidian_distances = euclidean_distances(cell_neighbor_hood)

                        neighbor_data: Dict = {"Cell": cellId}
                        for i, euclidian_distance in enumerate(euclidian_distances):
                            for j in range(neighbor_count + 1):
                                if j == 0:
                                    continue
                                neighbor_data[f"{j} Neighbor"] = euclidian_distances[0][j]

                        euclidean_distances_per_cell_data.append(neighbor_data)

                    # Create dfs
                    euclidean_distances_per_cell: pd.DataFrame = pd.DataFrame.from_records(
                        euclidean_distances_per_cell_data)

                    Reporter.upload_csv(data=euclidean_distances_per_cell,
                                        file_name=f"euclidean_distances",
                                        save_path=base_path, mlflow_folder=folder_name)

                    print("Imputing data")
                    imputed_cells: pd.DataFrame = KNNImputation.impute(train_data=train_data,
                                                                       test_data=replaced_test_data_cells,
                                                                       missing_values=0,
                                                                       n_neighbors=neighbor_count)

                    # Upload the imputed cells
                    Reporter.upload_csv(data=imputed_cells, save_path=base_path, file_name="imputed_cells",
                                        mlflow_folder=folder_name)

                    print("Evaluating r2 scores...")
                    r2_scores = KNNImputation.evaluate_performance(features=features,
                                                                   index_replacements=index_replacements,
                                                                   test_data=test_data,
                                                                   imputed_data=imputed_cells)

                    Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path,
                                              mlflow_folder=folder_name,
                                              prefix="imputed")

                    r2_scores["Origin"] = f"{run_option} {neighbor_count}"
                    combined_r2_score_data.append(r2_scores)

                    if neighbor_count == max_neighbors - 1:
                        nearest_neighbors_indices = nearest_neighbors_indices.T
                        nearest_neighbors_indices.rename(columns={0: "Base Cell"}, inplace=True)
                        nearest_neighbors_indices["Origin"] = f"{run_option} {neighbor_count}"
                        combined_nearest_neighbor_indices_data.append(nearest_neighbors_indices)

                        phenotypes["Origin"] = f"{run_option} {neighbor_count}"
                        combined_mapped_phenotype_data.append(phenotypes)

                        imputed_cells["Origin"] = f"{run_option} {neighbor_count}"
                        combined_imputed_data.append(imputed_cells)

                        replaced_test_data_cells["Origin"] = f"{run_option} {neighbor_count}"
                        combined_replaced_cell_data.append(replaced_test_data_cells)

                        euclidean_distances_per_cell["Origin"] = f"{run_option} {neighbor_count}"
                        combined_euclidean_distances_data.append(euclidean_distances_per_cell)

            combined_r2_scores = pd.concat([scores for scores in combined_r2_score_data])
            combined_nearest_neighbors = pd.concat(
                [nearest_neighbors for nearest_neighbors in combined_nearest_neighbor_indices_data])
            combined_mapped_phenotypes = pd.concat([phenotype for phenotype in combined_mapped_phenotype_data])
            combined_imputed_cells = pd.concat([imputed_data for imputed_data in combined_imputed_data])
            combined_replaced_data = pd.concat([replaced_cells for replaced_cells in combined_replaced_cell_data])
            combined_euclidean_distances = pd.concat(
                [euclidian_distances for euclidian_distances in combined_euclidean_distances_data])

            Reporter.upload_csv(data=combined_r2_scores, file_name="combined_r2_scores", save_path=base_path)
            Reporter.upload_csv(data=combined_nearest_neighbors, file_name="combined_nearest_neighbors",
                                save_path=base_path)
            Reporter.upload_csv(data=combined_mapped_phenotypes, file_name="combined_mapped_phenotypes",
                                save_path=base_path)
            Reporter.upload_csv(data=combined_imputed_cells, file_name="combined_imputed_data",
                                save_path=base_path)
            Reporter.upload_csv(data=combined_replaced_data, file_name="combined_replaced_data", save_path=base_path)
            Reporter.upload_csv(data=combined_euclidean_distances, file_name="combined_euclidean_distances",
                                save_path=base_path)

    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
