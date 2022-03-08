import mlflow
import logging
import pandas as pd
from pathlib import Path
from evaluation.evaluation import Evaluation
from folder_management.folder_management import FolderManagement
import argparse
from library.mlflow_helper.experiment_handler import ExperimentHandler

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=True,
                        help="The name of the experiment which should be evaluated",
                        type=str)
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run",
                        type=str)

    return parser.parse_args()


class ExperimentComparer:
    # All runs to compare
    runs: list = []
    client = mlflow.tracking.MlflowClient()
    base_path = Path("LatentSpaceExplorer/tmp")
    # The user given experiment name
    experiment_name: str
    # The experiment id
    experiment_id: str
    experiment_handler: ExperimentHandler

    def __init__(self, experiment_name: str):
        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        self.experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

        self.experiment_name = experiment_name
        self.experiment_id = self.experiment_handler.get_experiment_id_by_name(experiment_name=self.experiment_name,
                                                                               experiment_description="")

    def start_comparison(self):
        if self.experiment_id is None:
            print(f"Could not find an experiment with the given name: {self.experiment_name}")
            return

        self.base_path = FolderManagement.create_directory(self.base_path)

        with mlflow.start_run(run_name=args.run, experiment_id=self.experiment_id) as run:
            # Collect all experiments based on the search tag
            self.runs = ExperimentHandler.get_runs(self.experiment_id)

            if len(self.runs) == 0:
                print(f"No runs found.")
                print("Resources are being cleaned up.")
                # self.client.delete_run(run.info.run_id)
                return

            print(f"Found {len(self.runs)} runs.")

            self.experiment_handler.download_artifacts(self.base_path, self.runs)
            ae_scores: pd.DataFrame = self.experiment_handler.load_r2_scores_for_model(self.base_path, "AE")
            vae_scores: pd.DataFrame = self.experiment_handler.load_r2_scores_for_model(self.base_path, "VAE")

            mlflow.log_param("AE_marker_count", ae_scores.shape[0])
            mlflow.log_param("VAE_marker_count", vae_scores.shape[0])

            evaluation = Evaluation(self.base_path)
            evaluation.r2_scores_mean_values(ae_scores=ae_scores, vae_scores=vae_scores)

            self.__log_information()

    def __log_information(self):
        mlflow.log_param("Included Runs", len(self.runs))
        mlflow.log_param("Used Run Ids",
                         [x.info.run_id for x in self.runs])


if __name__ == "__main__":
    args = get_args()
    comparer = ExperimentComparer(args.experiment)
else:
    raise "Tool is meant to be executed as standalone"
