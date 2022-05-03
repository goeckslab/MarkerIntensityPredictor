import argparse
import pandas as pd
from library.plotting.plots import Plotting
from library.mlflow_helper.experiment_handler import ExperimentHandler
import mlflow
from pathlib import Path
import time
from library.data.folder_management import FolderManagement
from library.mlflow_helper.run_handler import RunHandler
from mlflow.entities import Run
from library.data.data_loader import DataLoader
from library.evalation.evaluation import Evaluation

base_path = "compare_r2_runs"


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be evaluated",
                        default="Default", type=str)
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--runs", nargs='+', action="store", required=True,
                        help="The run ids to compare", type=str)
    parser.add_argument("--run_names", nargs='+', action="store", required=True,
                        help="The run names to use for labeling purposes", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

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

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run) as run:

            runs_to_download: list = []
            for run_id in args.runs:
                run_to_download: Run = run_handler.get_run_by_id(experiment_id=associated_experiment_id, run_id=run_id)
                if run_to_download is not None:
                    runs_to_download.append(run_to_download)

            # Download all run data
            directory_for_run: dict = experiment_handler.download_artifacts(base_save_path=base_path,
                                                                            runs=runs_to_download)

            r2_scores: dict = {}
            features: list = []
            for i, (run_id, directory) in enumerate(directory_for_run.items()):

                r2_score: pd.DataFrame = DataLoader.load_r2_score(load_path=directory, file_name="imputed_r2_score.csv")

                run_name: str = run_handler.get_run_name_by_run_id(run_id=run_id, runs=runs_to_download)
                if r2_score is not None:
                    r2_scores[args.run_names[i]] = r2_score

                features = DataLoader.load_file(load_path=directory, file_name="Features.csv")["Features"].tolist()

            plotter: Plotting = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores=r2_scores, file_name="R2 Comparison")

            absolute_r2_score_performance: dict = {
                "Imputation Performance": Evaluation.create_absolute_score_performance(r2_scores=r2_scores,
                                                                                       features=features)}

            plotter.r2_scores_absolute_performance(absolute_score_performance=absolute_r2_score_performance,
                                                   file_name="Absolute Comparison")





    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
