# Compares the cell distances used by the KNN imputer between included spatial data or not
import mlflow
from pathlib import Path
import time
import argparse
import pandas as pd
from library.preprocessing.replacements import Replacer
from library.plotting.plots import Plotting
from sklearn.metrics.pairwise import nan_euclidean_distances, euclidean_distances
import numpy as np
from matplotlib.cbook import boxplot_stats
from typing import Dict, List
from scipy import stats
from library import DataLoader, FolderManagement, PhenotypeMapper, TTest, Preprocessing, Reporter, ExperimentHandler
import os

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
    parser.add_argument("--phenotypes", "-ph", action="store", required=False, help="The phenotype association")

    return parser.parse_args()


def process_phenotyping(args):
    phenotypes: pd.DataFrame = pd.read_csv(args.phenotypes)
    cells: pd.DataFrame = DataLoader.load_single_cell_data(file_name=args.file, keep_spatial=True, return_df=True)

    phenotype_spatial_data = pd.DataFrame(columns=["X", "Y", "Phenotype"])
    phenotype_spatial_data["X"] = cells["X_centroid"]
    phenotype_spatial_data["Y"] = cells["Y_centroid"]
    phenotype_spatial_data["Phenotype"] = phenotypes["phenotype"]

    keys = phenotypes["phenotype"].value_counts().keys().tolist()
    counts = phenotypes["phenotype"].value_counts().tolist()

    unique_phenotypes = pd.DataFrame(columns=keys)
    unique_phenotypes.loc[len(unique_phenotypes)] = counts
    Reporter.upload_csv(data=unique_phenotypes, save_path=base_path,
                        file_name="unique_phenotype_count")

    # Map the phenotypes to the indexes
    phenotype_mapping: pd.DataFrame = PhenotypeMapper.map_nn_to_phenotype(
        nearest_neighbors=nearest_neighbors_indices, phenotypes=phenotypes)

    phenotype_mapping = phenotype_mapping.T
    phenotype_mapping.rename(columns={1: "First Neighbor", 2: "Second Neighbor"}, inplace=True)

    Reporter.upload_csv(data=phenotype_mapping, save_path=base_path,
                        file_name="mapped_phenotypes")

    plotter.scatter_plot(data=phenotype_spatial_data, x="X", y="Y", hue="Phenotype", file_name="Phenotype Localization",
                         title="Phenotype Localization", marker='')


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

            # The euclidean distances for each run. Spatial and No Spatial
            euclidean_distances_all_cells: Dict = {}
            micron_distances_all_cells: Dict = {}

            # The euclidean distances for each cell, just in a different setup
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
                    train_data: pd.DataFrame = Preprocessing.normalize(data=train_cells.copy(), columns=features,
                                                                       create_dataframe=True)

                    # Load the test dataset
                    test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file,
                                                                     keep_spatial=use_spatial_information)

                    # Normalize test data set
                    test_data: pd.DataFrame = Preprocessing.normalize(data=test_cells.copy(), columns=features,
                                                                      create_dataframe=True)

                    # Replace values and return indices
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

                    if args.phenotypes is not None:
                        process_phenotyping(args)

                    # Save data
                    euclidean_distances_per_cell_data: List = []
                    micron_distances_per_cell_data: List = []
                    distances_per_neighbor_data: List = []

                    # Load dataframe again to include x and y data
                    if run_option == "No Spatial":
                        test_cells, features = DataLoader.load_single_cell_data(file_name=args.file,
                                                                                keep_spatial=True)
                        # Normalize test data set
                        test_data: pd.DataFrame = Preprocessing.normalize(data=test_cells.copy(), columns=features,
                                                                          create_dataframe=True)

                    for cellId, nearestNeighbors in nearest_neighbors_indices.iteritems():
                        cell = test_data.iloc[[cellId]][["X_centroid", "Y_centroid"]].reset_index(drop=True)
                        first_neighbor = test_data.iloc[[nearestNeighbors.iloc[1]]][
                            ["X_centroid", "Y_centroid"]].reset_index(drop=True)
                        second_neighbor = test_data.iloc[[nearestNeighbors.iloc[2]]][
                            ["X_centroid", "Y_centroid"]].reset_index(drop=True)

                        frames = [cell, first_neighbor, second_neighbor]
                        new_df = pd.concat(frames)
                        euclidian_distances = euclidean_distances(new_df)

                        euclidean_distances_per_cell_data.append({
                            "Cell": cellId,
                            "First Neighbor": euclidian_distances[0][1],
                            "Second Neighbor": euclidian_distances[0][2],
                        })

                        distances_per_neighbor_data.append({
                            "Neighbor": "First",
                            "Distance": euclidian_distances[0][1]
                        })

                        distances_per_neighbor_data.append({
                            "Neighbor": "Second",
                            "Distance": euclidian_distances[0][2]
                        })

                    # Create dfs
                    euclidean_distances_per_cell: pd.DataFrame = pd.DataFrame.from_records(
                        euclidean_distances_per_cell_data)
                    micron_distances_per_cell: pd.DataFrame = pd.DataFrame()
                    distances_per_neighbor = pd.DataFrame.from_records(distances_per_neighbor_data)

                    plotter.scatter_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                                         title="Spatial Distances", file_name="Spatial Distances")
                    plotter.joint_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                                       file_name="Distances of cells")
                    plotter.joint_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                                       file_name="KDE Distances of cells", kind="kde")

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

            outlier_count_data: List = []
            for key, distances in run_distances_per_neighbor.items():
                outlier_count_data.append({"Combination": f"{key} First", "Count": int(
                    len([y for stat in boxplot_stats(distances.loc[distances["Neighbor"] == 'First']['Distance']) for y
                         in stat['fliers']]))})

                outlier_count_data.append({
                    "Combination": f"{key} Second",
                    "Count": int(len(
                        [y for stat in boxplot_stats(distances.loc[distances["Neighbor"] == 'Second']['Distance']) for y
                         in stat['fliers']]))
                })

            outlier_count: pd.DataFrame = pd.DataFrame.from_records(outlier_count_data)

            # first_neighbor_t_test = stats.ttest_ind(euclidean_distances_all_cells[run_options[0]]["First Neighbor"],
            #                                        euclidean_distances_all_cells[run_options[1]]["First Neighbor"])
            # second_neighbor_t_test = stats.ttest_ind(euclidean_distances_all_cells[run_options[0]]["Second Neighbor"],
            #                        euclidean_distances_all_cells[run_options[1]]["Second Neighbor"])

            # t_test_data: List = [{
            #    "Neighbor": "First",
            #    "T-Value": first_neighbor_t_test[0],
            #    "p-Value": first_neighbor_t_test[1]
            #
            #           }, {
            #              "Neighbor": "Second",
            #             "T-Value": second_neighbor_t_test[0],
            #            "p-Value": second_neighbor_t_test[1]
            #
            #           }]

            t_test_data: List = [TTest.calculateTTest(euclidean_distances_all_cells[run_options[0]]["First Neighbor"],
                                                      euclidean_distances_all_cells[run_options[1]]["First Neighbor"],
                                                      column="Neighbor", column_value="First Neighbor"),
                                 TTest.calculateTTest(euclidean_distances_all_cells[run_options[0]]["Second Neighbor"],
                                                      euclidean_distances_all_cells[run_options[1]]["Second Neighbor"],
                                                      column="Neighbor", column_value="Second Neighbor")]

            t_test: pd.DataFrame = pd.concat(t_test_data)
            Reporter.upload_csv(data=outlier_count, save_path=base_path, file_name="outlier_count")
            Reporter.upload_csv(data=t_test, save_path=base_path, file_name="t_test_data")
            plotter.box_plot(data=run_distances_per_neighbor, x="Neighbor", y="Distance",
                             title="Neighbor Spatial Distances", file_name="Euclidean Distances")
            plotter.dist_plot(data=euclidean_distances_per_cell, x="First Neighbor",
                              file_name="First Neighbor Distance distribution")

            plotter.dist_plot(data=euclidean_distances_per_cell, x="Second Neighbor",
                              file_name="Second Neighbor Distance distribution")

            plotter.joint_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                               file_name="Distances of cells")
            plotter.joint_plot(data=euclidean_distances_per_cell, x="First Neighbor", y="Second Neighbor",
                               file_name="KDE Distances of cells", kind="kde")



    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
