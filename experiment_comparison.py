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
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")

    return parser.parse_args()


if __name__ == "__main__":
    print("Experiment evaluation started...")
    args = get_args()

    if len(args.experiments) < 2:
        raise ValueError("Please specify at least two experiments for comparison")

    FolderManagement.create_directory(base_path)

    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        experiment_handler = ExperimentHandler(client=client)

        vae_r2_mean_scores: {} = {}
        vae_r2_combined_scores: {} = {}

        en_r2_mean_scores: {} = {}
        en_r2_combined_scores: {} = {}

        ae_r2_mean_scores: {} = {}
        ae_r2_combined_scores: {} = {}

        for experiment_name in args.experiments:
            experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                              create_experiment=False)

            if experiment_id is None:
                print(f"Could not find experiment with name {experiment_name}. Skipping..")
                continue

            run: Run = experiment_handler.get_run_comparison_run(experiment_id=experiment_id)

            if run is None:
                print("Could not found a run labeled 'Run Comparison'. "
                      f"Please create a run by running the run_comparison.py for the experiment {args.experiments[0]}.")
                continue

            # Download artifacts
            experiment_handler.download_artifacts(base_path, run=run)

            vae_r2_mean_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="vae_mean_r2_score.csv")
            vae_r2_combined_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="vae_combined_r2_score.csv")

            en_r2_mean_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="en_mean_r2_score.csv")
            en_r2_combined_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="en_combined_r2_score.csv")

            ae_r2_mean_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="ae_mean_r2_score.csv")
            ae_r2_combined_scores[experiment_name] = DataLoader.load_file(
                load_path=Path(base_path, run.info.run_id),
                file_name="ae_combined_r2_score.csv")

        # The new experiment which is used to store the evaluation data
        evaluation_experiment_id: str = experiment_handler.create_experiment("Experiment Comparison Test",
                                                                             "Evaluation of multiple experiments wll be listed here")

        # Create experiment and evaluation data
        with mlflow.start_run(experiment_id=evaluation_experiment_id,
                              run_name=args.run) as run:
            plotting: Plotting = Plotting(base_path=base_path, args=args)
            plotting.r2_scores_distribution(r2_scores=vae_r2_combined_scores, file_name="VAE R2 Score Distribution")
            plotting.r2_scores_distribution(r2_scores=en_r2_combined_scores, file_name="EN R2 Score Distribution")
            plotting.r2_scores_distribution(r2_scores=ae_r2_combined_scores, file_name="AE R2 Score Distribution")
            plotting.r2_score_model_distribution(vae_r2_scores=vae_r2_combined_scores,
                                                 ae_r2_scores=ae_r2_combined_scores,
                                                 en_r2_scores=en_r2_combined_scores,
                                                 file_name="VAE vs. AE vs. EN Score Distribution")
            plotting.r2_score_model_distribution_vae_vs_en(vae_r2_scores=vae_r2_combined_scores,
                                                           en_r2_scores=en_r2_combined_scores,
                                                           file_name="VAE vs. EN Score Distribution")

            plotting.r2_score_model_mean(vae_r2_scores=vae_r2_mean_scores, ae_r2_scores=ae_r2_mean_scores,
                                         en_r2_scores=en_r2_mean_scores, file_name="Mean R2 score comparison")

    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
