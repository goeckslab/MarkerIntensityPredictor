import mlflow
from mlflow.tracking.fluent import _get_experiment_id
import logging
from data_management.data_management import DataManagement
import pandas as pd
from pathlib import Path
from evaluation.evaluation import Evaluation
from folder_management.folder_management import FolderManagement

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class ExperimentComparer:
    # All Experiments to compare
    experiments: list = []
    client = mlflow.tracking.MlflowClient()
    base_path = Path("tmp")

    def __init__(self, search_tag: str = None):
        self.base_path = FolderManagement.create_directory(self.base_path)
        with mlflow.start_run(run_name="ae_vae_experiment_comparison") as run:
            self.__get_experiments(search_tag)
            data_manager = DataManagement(self.base_path)
            data_manager.download_artifacts(self.experiments)
            ae_scores: pd.DataFrame = data_manager.load_r2_scores_for_model("AE")
            vae_scores: pd.DataFrame = data_manager.load_r2_scores_for_model("VAE")

            evaluation = Evaluation(self.base_path)
            evaluation.r2_scores_mean_values(ae_scores=ae_scores, vae_scores=vae_scores)

    def __get_experiments(self, search_tag: str):
        experiment_id = _get_experiment_id()

        all_run_infos = reversed(self.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run = self.client.get_run(run_info.run_id)

            if search_tag is None:
                self.experiments.append(full_run)
            else:
                if "Group" in full_run.data.tags:
                    if full_run.data.tags.get("Group") == search_tag:
                        self.experiments.append(full_run)

        experiments_found: int = 0
        for experiment in self.experiments:
            if 'mlflow.parentRunId' not in experiment.data.tags:
                experiments_found += 1

        print(f"Found {experiments_found} experiments.")


if __name__ == "__main__":
    comparer = ExperimentComparer("Mean")
else:
    raise "Tool is meant to be executed as standalone"
