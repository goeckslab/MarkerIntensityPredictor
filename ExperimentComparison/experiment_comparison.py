import mlflow
from mlflow.tracking.fluent import _get_experiment_id
import logging
from data_management.data_management import DataManagement
import pandas as pd
from pathlib import Path
from evaluation.evaluation import Evaluation
from folder_management.folder_management import FolderManagement
from args import ArgumentParser
from experiment_handler.experiment_handler import ExperimentHandler

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class ExperimentComparer:
    # All runs to compare
    runs: list = []
    client = mlflow.tracking.MlflowClient()
    base_path = Path("tmp")
    # The user given experiment name
    experiment_name: str
    # The experiment id
    experiment_id: str

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_id = ExperimentHandler.get_experiment_id_by_name(self.experiment_name)

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
                self.client.delete_run(run.info.run_id)
                return

            self.__info_output()

            data_manager = DataManagement(self.base_path)
            data_manager.download_artifacts(self.runs)
            ae_scores: pd.DataFrame = data_manager.load_r2_scores_for_model("AE")
            vae_scores: pd.DataFrame = data_manager.load_r2_scores_for_model("VAE")

            evaluation = Evaluation(self.base_path)
            evaluation.r2_scores_mean_values(ae_scores=ae_scores, vae_scores=vae_scores)

    def __info_output(self):
        print(f"Found {len(self.runs)} runs.")
        mlflow.log_param("Included Runs", len(self.runs))
        mlflow.log_param("Used Run Ids",
                         [x.info.run_id for x in self.runs])


if __name__ == "__main__":
    args = ArgumentParser.get_args()
    comparer = ExperimentComparer(args.experiment)
else:
    raise "Tool is meant to be executed as standalone"
