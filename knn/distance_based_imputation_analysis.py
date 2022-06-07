import sys, os
import mlflow
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import ExperimentHandler, RunHandler, FolderManagement, DataLoader, Reporter, Plotting, \
    KNNDataAnalysisPlotting
from pathlib import Path
import argparse

results_folder = Path("imputation_analysis")


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


if __name__ == '__main__':
    args = get_args()
    run_name = f"KNN Imputation Data Analysis {args.percentage}"
    experiment_handler: ExperimentHandler = ExperimentHandler(tracking_url=args.tracking_url)
    run_handler: RunHandler = RunHandler(tracking_url=args.tracking_url)

    experiment_name = args.experiment

    # Get source experiment
    source_run_name: str = f"KNN Distance Based Data Imputation Percentage {args.percentage}"
    source_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    FolderManagement.create_directory(path=results_folder)

    try:

        # The id of the associated
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                                create_experiment=True)
        # Set experiment id
        mlflow.set_experiment(experiment_id=associated_experiment_id)

        runs_in_source_experiment: Dict = run_handler.get_run_and_child_runs(experiment_id=source_experiment_id,
                                                                             run_name=source_run_name)

        runs: List = [run for run in runs_in_source_experiment.keys()]

        # Download data
        download_paths: Dict = run_handler.download_artifacts(base_save_path=results_folder, runs=runs)

        # Instantiate plotter
        plotter: Plotting = Plotting(base_path=results_folder, args=args)
        data_plotter: KNNDataAnalysisPlotting = KNNDataAnalysisPlotting(base_path=results_folder)

        # Delete previous run
        run_handler.delete_runs_and_child_runs(experiment_id=source_experiment_id, run_name=run_name)

        # Get source experiment
        source_run_name: str = f"KNN Distance Based Data Imputation Percentage {args.percentage}"
        source_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:

            imputation_scores: pd.DataFrame = DataLoader.load_file(load_path=download_paths[0],
                                                                   file_name="imputed_r2_scores.csv")
            plotter.bar_plot(data=imputation_scores, x="Marker", y="Score", hue="Distance", file_name="R2 Scores",
                             title="R2 Scores")


    except:
        raise

    finally:
        FolderManagement.delete_directory(results_folder)
