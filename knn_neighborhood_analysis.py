from library import ExperimentHandler, RunHandler, Reporter, FolderManagement, DataLoader, Plotting, \
    KNNDataAnalysisPlotting
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import mlflow
from typing import List, Dict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

base_path = Path("knn_neighbor_hood_analysis")


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

    parser.add_argument("--phenotypes", "-ph", action="store", required=False, help="The phenotype association")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run_name = "KNN Neighbor Data Analysis"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    experiment_name = args.experiment

    # The id of the associated
    associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                            create_experiment=True)
    # Set experiment id
    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(path=base_path)

    try:
        # Get source experiment
        source_run_name: str = f"KNN Neighborhood Data Generation Percentage {args.percentage}"
        source_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

        runs_in_source_experiment: Dict = run_handler.get_run_and_child_runs(experiment_id=source_experiment_id,
                                                                             run_name=source_run_name)

        runs: List = [run for run in runs_in_source_experiment.keys()]

        # Download data
        download_paths: Dict = run_handler.download_artifacts(base_save_path=base_path, runs=runs)

        # Instantiate plotter
        plotter: Plotting = Plotting(base_path=base_path, args=args)
        data_plotter: KNNDataAnalysisPlotting = KNNDataAnalysisPlotting(base_path=base_path)

        # Delete previous run
        run_handler.delete_runs_and_child_runs(experiment_id=source_experiment_id, run_name=run_name)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:
            phenotype_associations_for_neighbors: pd.DataFrame = DataLoader.load_file(
                load_path=list(download_paths.values())[0],
                file_name="combined_phenotype_neighbors.csv")

            # Load phenotype associations
            cell_phenotypes: pd.DataFrame = DataLoader.load_file(load_path=args.phenotypes)

            # Combine phenotypes with neighbor data
            spatial: pd.DataFrame = phenotype_associations_for_neighbors[
                phenotype_associations_for_neighbors["Mode"] == "Spatial"].reset_index(drop=True)
            spatial["Origin"] = cell_phenotypes["phenotype"]

            # Combine phenotypes with neighbor data
            non_spatial = phenotype_associations_for_neighbors[
                phenotype_associations_for_neighbors["Mode"] == "No Spatial"].reset_index(drop=True)
            non_spatial["Origin"] = cell_phenotypes["phenotype"]

            colors = {'Luminal': 'red',
                      'Basal': 'green',
                      'Immune': 'blue'}

            for phenotype in spatial["Origin"].unique():
                data_plotter.pie_chart(
                    keys=spatial[spatial["Origin"] == phenotype]["First Neighbor"].unique(),
                    data=spatial[spatial["Origin"] == phenotype]["First Neighbor"].value_counts().to_list(),
                    file_name=f"Spatial First Neighbor {phenotype} Distribution", colors=colors,
                    title=f"Phenotype of nearest neighbor for {phenotype} cells including spatial data")

                data_plotter.pie_chart(keys=spatial[spatial["Origin"] == phenotype]["Second Neighbor"].unique(),
                                       data=spatial[spatial["Origin"] == phenotype][
                                           "Second Neighbor"].value_counts().to_list(),
                                       file_name=f"Spatial Second Neighbor {phenotype} Distribution", colors=colors,
                                       title=f"Phenotype of nearest neighbor for {phenotype} cells including spatial data")

                for phenotype in non_spatial["Origin"].unique():
                    data_plotter.pie_chart(
                        keys=non_spatial[non_spatial["Origin"] == phenotype]["First Neighbor"].unique(),
                        data=non_spatial[non_spatial["Origin"] == phenotype]["First Neighbor"].value_counts().to_list(),
                        file_name=f"No Spatial First Neighbor {phenotype} Distribution", colors=colors,
                        title=f"Phenotype of nearest neighbor for {phenotype} cells without spatial data")

                data_plotter.pie_chart(keys=non_spatial[non_spatial["Origin"] == phenotype]["Second Neighbor"].unique(),
                                       data=non_spatial[non_spatial["Origin"] == phenotype][
                                           "Second Neighbor"].value_counts().to_list(),
                                       file_name=f"No Spatial Second Neighbor {phenotype} Distribution",
                                       colors=colors,
                                       title=f"Phenotype of nearest neighbor for {phenotype} cells without spatial data")


    except:
        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)
    finally:
        FolderManagement.delete_directory(path=base_path)
