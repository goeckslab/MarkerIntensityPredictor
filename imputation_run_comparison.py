from pathlib import Path
import argparse
from typing import Optional
import mlflow
import pandas as pd
from library.mlflow_helper.experiment_handler import ExperimentHandler
from mlflow.entities import Run
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
from library.plotting.plots import Plotting
from library.mlflow_helper.run_handler import RunHandler
from library.evalation.evaluation import Evaluation

base_path = Path("imputation_score_comparison")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run",
                        type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="Assigns the run to a particular experiment. "
                             "If the experiment does not exists it will create a new one.",
                        default="Default", type=str)
    parser.add_argument("--runs", action="store", nargs='+', required=False,
                        help="The runs which should be compared", type=str)
    parser.add_argument("--percentage", "-p", help="The percentage which is compared", required=True, type=float)
    parser.add_argument("--labels", "-l", nargs='+', help="The labels which should be added", required=False, type=str)
    parser.add_argument("--parent_run", "-pr", help="The parent run id", required=True, type=str)

    return parser.parse_args()


def get_associated_experiment_id(args) -> Optional[str]:
    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    return associated_experiment_id


if __name__ == "__main__":
    args = get_args()
    FolderManagement.create_directory(base_path)

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    associated_experiment_id: str = get_associated_experiment_id(args)
    try:
        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run.strip()) as run:
            mlflow.log_param("Runs used", args.runs)

            runs: [] = []
            for run_name in args.runs:
                run_name = run_name.strip()
                run: Run = run_handler.get_run_by_name(experiment_id=associated_experiment_id,
                                                       run_name=run_name, parent_run_id=args.parent_run)
                if run is None:
                    continue
                runs.append(run)

            run_directories: [] = run_handler.download_artifacts(runs=runs, base_save_path=base_path,
                                                                 mlflow_folder="")

            imputed_r2_scores: pd.DataFrame = pd.DataFrame()
            frames = []
            r2_scores: dict = {}
            features: list = []
            for directory in run_directories:
                print("Loading files...")
                data: pd.DataFrame = DataLoader.load_file(load_path=directory, file_name="imputed_r2_score.csv")

                if data is None:
                    continue

                run_name: str = run_handler.get_run_name_by_run_id(run_id=directory.stem, runs=runs)

                if run_name is None:
                    raise ValueError(f"Run name for id {directory.stem} is none")

                r2_scores[run_name] = data

                features = DataLoader.load_file(load_path=directory, file_name="Features.csv")["Features"].tolist()

            plotter: Plotting = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores=r2_scores, file_name=f"Imputation Performance {args.percentage}")

            absolute_r2_score_performance: dict = {
                "Imputation Performance": Evaluation.create_absolute_score_performance(r2_scores=r2_scores,
                                                                                       features=features)}

            plotter.r2_scores_combined_bar_plot(r2_scores=absolute_r2_score_performance,
                                                file_name=f"Absolute Performance {args.percentage}",
                                                legend_labels=args.labels)

    except BaseException as ex:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
