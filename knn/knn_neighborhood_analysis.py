import os, sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import ExperimentHandler, RunHandler, Reporter, Plotting, \
    KNNDataAnalysisPlotting, DataLoader, FolderManagement, PhenotypeMapper, Evaluation
import pandas as pd
from pathlib import Path
import argparse
import mlflow
from typing import List, Dict

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


def translate_key(key: str):
    if "no_spatial" in key:
        new_key = f"No Spatial {key.split('_')[-1]}"

    else:
        new_key = f"Spatial {key.split('_')[-1]}"

    return new_key


if __name__ == "__main__":
    args = get_args()

    run_name = f"KNN Neighbor Data Analysis {args.percentage}"

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
            # Load phenotype associations
            cell_phenotypes: pd.DataFrame = DataLoader.load_file(load_path=args.phenotypes)

            loaded_frames: Dict = DataLoader.load_files(load_path=list(download_paths.values())[0],
                                                        file_name="imputed_r2_score.csv")

            frames: List = []
            value: pd.DataFrame
            for key, value in loaded_frames.items():
                value["Origin"] = translate_key(key)
                value["Model"] = translate_key(key)

                if "Marker" in value:
                    value.rename(columns={"Marker": "Feature"}, inplace=True)

                frames.append(value)

            imputed_scores: pd.DataFrame = pd.concat(frames, axis=0)

            features = imputed_scores["Feature"].unique()

            spatial_performance: Dict = {}
            non_spatial_performance: Dict = {}
            for key in loaded_frames.keys():
                if "no_spatial" in key:
                    non_spatial_performance[key] = loaded_frames.get(key)

                else:
                    spatial_performance[key] = loaded_frames.get(key)

            absolute_r2_score_performance: Dict = {
                "Spatial Imputation Performance": Evaluation.create_absolute_score_performance(
                    r2_scores=spatial_performance,
                    features=features)}

            plotter.r2_scores_combined_bar_plot(r2_scores=absolute_r2_score_performance, file_name="Spatial R2 Scores",
                                                features=features)

            absolute_r2_score_performance: dict = {
                "No Spatial Imputation Performance": Evaluation.create_absolute_score_performance(
                    r2_scores=non_spatial_performance,
                    features=features)}
            plotter.r2_scores_combined_bar_plot(r2_scores=absolute_r2_score_performance,
                                                file_name="Non Spatial R2 Scores",
                                                features=features)

            # Load nn indices
            loaded_frames = DataLoader.load_files(load_path=list(download_paths.values())[0],
                                                  file_name="nearest_neighbor_indices.csv")
            frames: List = []
            for key, value in loaded_frames.items():
                value["Origin"] = translate_key(key)
                value["Neighbor Count"] = key.split('_')[-1]
                frames.append(value)

            nearest_neighbor_indices: pd.DataFrame = pd.concat(frames)

            print(nearest_neighbor_indices)
            phenotype_labeling: Dict = {}
            for origin in tqdm(nearest_neighbor_indices["Origin"].unique()):
                indexes = nearest_neighbor_indices[nearest_neighbor_indices["Origin"] == origin]
                neighbor_count = int(indexes.at[0, "Neighbor Count"])

                cleaned_df = indexes.drop(columns=["Origin", "Neighbor Count"])
                phenotypes = PhenotypeMapper.map_nn_to_phenotype(nearest_neighbors=cleaned_df,
                                                                 phenotypes=cell_phenotypes,
                                                                 neighbor_count=neighbor_count)

                phenotype_labeling[origin] = phenotypes

            colors = {'Luminal': 'red',
                      'Basal': 'green',
                      'Immune': 'blue'}

            for origin in phenotype_labeling.keys():
                spatial = "no spatial" if origin == "no spatial" else "spatial"
                head_spatial = "No Spatial" if origin == "no spatial" else "Spatial"
                data = phenotype_labeling.get(origin).T
                data["Base Cell"] = cell_phenotypes["phenotype"].values

                Reporter.upload_csv(data=data, save_path=base_path, file_name=f"{origin}_phenotypes")

                for phenotype in cell_phenotypes["phenotype"].unique():

                    for neighbor in data.columns:
                        data_plotter.pie_chart(
                            keys=data[data["Base Cell"] == phenotype][neighbor].unique(),
                            data=data[data["Base Cell"] == phenotype][neighbor].value_counts().to_list(),
                            file_name=f"{head_spatial} {neighbor} Neighbor {phenotype} Distribution", colors=colors,
                            title=f"Phenotype of nearest neighbor for {phenotype} cells including {spatial} data")

    except:
        raise
    finally:
        print("Analysis done")
        FolderManagement.delete_directory(path=base_path)
