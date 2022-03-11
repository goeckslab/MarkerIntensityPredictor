import logging
import argparse
from library.plotting.plots import Plotting
import pandas as pd
from library.mlflow_helper.experiment_handler import ExperimentHandler
from pathlib import Path
import mlflow
from mlflow.entities import Run, Experiment
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

base_path = Path("experiment_comparison")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", "-e", nargs='+', action="store", required=True,
                        help="The name of the experiments which should be evaluated", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    experiment_name = args.experiments[0]
    comp_experiment_name = args.experiments[1]

    FolderManagement.create_directory(base_path)

    if len(args.experiments) != 2:
        raise ValueError("Please specify at least two experiments for comparison")

    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler = ExperimentHandler(client=client)
    exp_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                               create_experiment=False)
    comparison_exp_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=comp_experiment_name,
                                                                          create_experiment=False)

    if exp_id is None or comparison_exp_id is None:
        raise LookupError("Could not find all specified experiments. Please make sure they exist.")

    # Load runs
    run: Run = experiment_handler.get_run_comparison_run(experiment_id=exp_id)
    comparison_run: Run = experiment_handler.get_run_comparison_run(experiment_id=comparison_exp_id)

    if run is None:
        raise LookupError(
            "Could not found a run labeled 'Run Comparison'. "
            f"Please create a run by running the run_comparison.py for the experiment {args.experiments[0]}.")

    if comparison_run is None:
        raise LookupError(
            "Could not found a run labeled 'Run Comparison'. "
            f"Please create a run by running the run_comparison.py for the experiment {args.experiments[1]}.")

    experiment_handler.download_artifacts(base_path, run=run)
    experiment_handler.download_artifacts(base_path, run=comparison_run)

    vae_mean_score: pd.DataFrame = DataLoader.load_file(Path(base_path, run.info.run_id), "vae_mean_r2_score.csv")
    comp_vae_mean_scores: pd.DataFrame = DataLoader.load_file(Path(base_path, comparison_run.info.run_id),
                                                              "vae_mean_r2_score.csv")

    if vae_mean_score is None or comp_vae_mean_scores is None:
        raise ValueError("Could not find all requested r2 scores dataframes. Please make sure they exist")

    # The new experiment which is used to store the evaluation data
    evaluation_experiment_id: str = experiment_handler.create_experiment("Experiment Comparison",
                                                                         "Evaluation of multiple experiments wll be listed here")
    with mlflow.start_run(experiment_id=evaluation_experiment_id,
                          run_name=f"{experiment_name} {comp_experiment_name} Comparison") as run:
        plotting: Plotting = Plotting(base_path=base_path, args=args)
        plotting.compare_r2_scores(r2_scores=vae_mean_score, compare_score=comp_vae_mean_scores,
                                   r2_score_title=experiment_name, compare_title=comp_experiment_name,
                                   file_name=f"{experiment_name}_{comp_experiment_name}_comparison")
