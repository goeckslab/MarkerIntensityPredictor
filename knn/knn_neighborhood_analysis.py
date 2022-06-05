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


def plot_r2_scores(r2_scores: Dict, title: str, file_name: str):
    absolute_r2_score_performance: Dict = {
        title: Evaluation.create_absolute_score_performance(
            r2_scores=r2_scores,
            features=features)}

    plotter.r2_scores_combined_bar_plot(r2_scores=absolute_r2_score_performance, file_name=file_name,
                                        features=features)


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
                    non_spatial_performance[translate_key(key)] = loaded_frames.get(key)

                else:
                    spatial_performance[translate_key(key)] = loaded_frames.get(key)

            # Plot scores
            plot_r2_scores(r2_scores=spatial_performance, title="Spatial Imputation Performance",
                           file_name="Spatial R2 Scores")
            plot_r2_scores(r2_scores=non_spatial_performance, title="No Spatial Imputation Performance",
                           file_name="Non Spatial R2 Scores")

            # Load nn distances
            loaded_frames = DataLoader.load_files(load_path=list(download_paths.values())[0],
                                                  file_name="euclidean_distances.csv")

            frames: List = []
            for key, value in loaded_frames.items():
                value["Origin"] = translate_key(key)
                value["Neighbor Count"] = key.split('_')[-1]
                frames.append(value)

            euclidean_distances = pd.concat(frames)

            test_df = pd.DataFrame(columns=["Distance", "Neighbor", "Origin"])

            frames: List = []
            for origin in tqdm(euclidean_distances["Origin"].unique()):
                data: pd.DataFrame = euclidean_distances[euclidean_distances["Origin"] == origin].copy()
                data.drop([col for col in data.columns if 'Neighbor' not in col], axis=1, inplace=True)
                data.drop(columns=["Neighbor Count"], axis=1, inplace=True)
                converted_df = \
                    data.T.unstack().reset_index(level=1, name='Distance').rename(columns={'level_1': 'Neighbor'})[
                        ['Distance', 'Neighbor']]
                converted_df["Origin"] = origin
                converted_df["Spatial"] = "Y" if "No Spatial" not in origin else "N"
                converted_df["Neighbor Count"] = converted_df["Origin"].apply(lambda x: x.split(' ')[-1])
                frames.append(converted_df)

            neighbor_distances: pd.DataFrame = pd.concat(frames).dropna()

            no_spatial_neighbor_distances: pd.DataFrame = neighbor_distances[
                neighbor_distances["Spatial"] == "N"].copy()
            no_spatial_neighbor_distances.drop(columns=["Spatial"], inplace=True)
            no_spatial_neighbor_distances.reset_index(inplace=True, drop=True)
            max_neighbor_count: int = no_spatial_neighbor_distances["Neighbor Count"].max()
            no_spatial_neighbor_distances = no_spatial_neighbor_distances[
                no_spatial_neighbor_distances["Origin"] == f"No Spatial {max_neighbor_count}"]

            spatial_neighbor_distances: pd.DataFrame = neighbor_distances[
                neighbor_distances["Spatial"] == "Y"].copy()
            spatial_neighbor_distances.drop(columns=["Spatial"], inplace=True)
            spatial_neighbor_distances.reset_index(inplace=True, drop=True)
            max_neighbor_count: int = spatial_neighbor_distances["Neighbor Count"].max()
            spatial_neighbor_distances = spatial_neighbor_distances[
                spatial_neighbor_distances["Origin"] == f"Spatial {max_neighbor_count}"]

            max_neighbor_count: int = neighbor_distances["Neighbor Count"].max()

            cond_one = neighbor_distances["Origin"] == f"Spatial {max_neighbor_count}"
            cond_two = neighbor_distances["Origin"] == f"No Spatial {max_neighbor_count}"
            neighbor_distances_cleaned = neighbor_distances.loc[cond_one | cond_two]

            plotter.box_plot(data=neighbor_distances_cleaned, x="Neighbor", y="Distance", hue="Origin",
                             title="Distance distribution", file_name="Distance Distribution")

            plotter.box_plot(data=no_spatial_neighbor_distances, x="Neighbor", y="Distance",
                             title="No Spatial Distance distribution", file_name="No Spatial Distance Distribution",
                             show_outliers=True)

            plotter.box_plot(data=spatial_neighbor_distances, x="Neighbor", y="Distance",
                             title="Spatial Distance distribution", file_name="Spatial Distance Distribution",
                             show_outliers=False)
            frames: List = []
            for neighbor_count in neighbor_distances["Neighbor Count"].unique():
                spatial_data = neighbor_distances[(neighbor_distances["Neighbor Count"] == neighbor_count) & (
                        neighbor_distances["Spatial"] == "Y")]

                no_spatial_data = neighbor_distances[(neighbor_distances["Neighbor Count"] == neighbor_count) & (
                        neighbor_distances["Spatial"] == "N")]

                # Calculate mean difference distance
                for neighbor in spatial_data["Neighbor"].unique():
                    mean_difference = no_spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].mean() - \
                                      spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].mean()

                    median_difference = no_spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].median() - \
                                        spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].median()

                    frames.append({
                        "Neighbor": neighbor,
                        "Mean Difference": mean_difference,
                        "Median Difference": median_difference,
                        "Data": "Difference N. Spatial vs Spatial"
                    })

                # Calculate mean and median for no spatial
                for neighbor in no_spatial_data["Neighbor"].unique():
                    mean_difference = no_spatial_data[no_spatial_data["Neighbor"] == neighbor]["Distance"].mean()
                    median_difference = no_spatial_data[no_spatial_data["Neighbor"] == neighbor]["Distance"].median()

                    frames.append({
                        "Neighbor": neighbor,
                        "Mean Difference": mean_difference,
                        "Median Difference": median_difference,
                        "Data": "No Spatial"
                    })

                # Calculate mean and median for no spatial
                for neighbor in spatial_data["Neighbor"].unique():
                    mean_difference = spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].mean()
                    median_difference = spatial_data[spatial_data["Neighbor"] == neighbor]["Distance"].median()

                    frames.append({
                        "Neighbor": neighbor,
                        "Mean Difference": mean_difference,
                        "Median Difference": median_difference,
                        "Data": "Spatial"
                    })

            neighbor_differences: pd.DataFrame = pd.DataFrame.from_records(frames)
            neighbor_differences.drop_duplicates(inplace=True, subset=["Neighbor", "Data"], keep='first')
            neighbor_differences["Experiment"] = f"{args.experiment} {args.percentage}"

            # Upload csv
            Reporter.upload_csv(data=neighbor_differences, save_path=base_path,
                                file_name="Neighbor Distance Differences")

            neighbor_differences = pd.melt(neighbor_differences, id_vars=['Neighbor', "Data"],
                                           value_vars=['Mean Difference', 'Median Difference'], var_name='Metric',
                                           value_name='Distance')

            data_plotter.line_plot(data=neighbor_differences, x="Neighbor", y="Distance", hue="Metric",
                                   file_name="Metrics", title="Metrics", style="Data")

            # Load nn indices
            loaded_frames = DataLoader.load_files(load_path=list(download_paths.values())[0],
                                                  file_name="nearest_neighbor_indices.csv")
            frames: List = []
            for key, value in loaded_frames.items():
                value["Origin"] = translate_key(key)
                value["Neighbor Count"] = key.split('_')[-1]
                frames.append(value)

            nearest_neighbor_indices: pd.DataFrame = pd.concat(frames)

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

            max_neighbor_count: int = nearest_neighbor_indices["Neighbor Count"].max()

            spatial_phenotypes = phenotype_labeling.get(f"Spatial {max_neighbor_count}")
            spatial_phenotypes = spatial_phenotypes.T
            spatial_phenotypes["Base Cell"] = cell_phenotypes["phenotype"].values

            no_spatial_phenotypes = phenotype_labeling.get(f"No Spatial {max_neighbor_count}")
            no_spatial_phenotypes = no_spatial_phenotypes.T
            no_spatial_phenotypes["Base Cell"] = cell_phenotypes["phenotype"].values

            Reporter.upload_csv(data=spatial_phenotypes, save_path=base_path, file_name=f"spatial_phenotypes")
            Reporter.upload_csv(data=no_spatial_phenotypes, save_path=base_path, file_name=f"no_spatial_phenotypes")

            # Report value count for each phenotype for each neighbor
            Reporter.upload_csv(data=spatial_phenotypes.apply(lambda x: x.unique()), save_path=base_path,
                                file_name=f"spatial_phenotypes_value_counts")
            Reporter.upload_csv(data=no_spatial_phenotypes.apply(lambda x: x.unique()), save_path=base_path,
                                file_name=f"no_spatial_phenotypes_value_counts")

            keys = sorted(spatial_phenotypes["Base Cell"].unique())

            pie_chart_data: Dict = {}
            for column in spatial_phenotypes.columns:
                if column == "Base Cell":
                    continue
                pie_chart_data[column] = spatial_phenotypes[column]

            data_plotter.pie_chart(data=pie_chart_data, file_name="Spatial Phenotype Composition", keys=keys)

            keys = no_spatial_phenotypes["Base Cell"].unique()
            pie_chart_data: Dict = {}
            for column in no_spatial_phenotypes.columns:
                if column == "Base Cell":
                    continue

                pie_chart_data[column] = no_spatial_phenotypes[column]

            data_plotter.pie_chart(data=pie_chart_data, file_name="No Spatial Phenotype Composition", keys=keys)



    except:
        raise
    finally:
        print("Analysis done")
        FolderManagement.delete_directory(path=base_path)
