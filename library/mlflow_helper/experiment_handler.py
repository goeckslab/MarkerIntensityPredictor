import mlflow.exceptions
from mlflow.entities import Run, RunInfo
from pathlib import Path
from typing import Optional
from library.data.folder_management import FolderManagement
from mlflow.exceptions import ErrorCode


class ExperimentHandler:
    client = None

    # Cache for already downloaded runs
    __runs: dict = {}

    def __init__(self, client):
        self.client = client

    def get_experiment_id_by_name(self, experiment_name: str, experiment_description: str = None,
                                  create_experiment: bool = True) -> Optional[str]:
        """
        Gets the experiment id associated with the given experiment name.
        If no experiment is found by default a new experiment will be created
        @param experiment_name: The experiment name
        @param experiment_description: The description for a new experiment
        @param create_experiment: Should the experiment be created if it does not exist
        @return: The experiment id
        """

        # The experiment id
        found_experiment_id = None

        experiments = self.client.list_experiments()  # returns a list of mlflow.entities.Experiment
        for experiment in experiments:
            if experiment.name == experiment_name:
                found_experiment_id = experiment.experiment_id

        if found_experiment_id is None and create_experiment:
            found_experiment_id = self.create_experiment(name=experiment_name, description=experiment_description)

        return found_experiment_id

    def create_experiment(self, name: str, description: str = "") -> str:
        """
        Creates a new experiment with the given name
        @param name: The name of the experiment
        @param description: The description for the experiment
        @return: The string of the newly created experiment
        """
        try:
            experiment_id: str = self.client.create_experiment(name=name)
            self.client.set_experiment_tag(experiment_id, "description", description)
            return experiment_id

        except mlflow.exceptions.RestException as ex:
            # As experiment exist just return the existing experiment id
            if ex.error_code == ErrorCode.Name(mlflow.exceptions.RESOURCE_ALREADY_EXISTS):
                return self.get_experiment_id_by_name(experiment_name=name, create_experiment=False)

        except BaseException as ex:
            # print("Experiment does already exists. Deleting and recreating experiment.")
            # experiment_id = self.client.get_experiment_by_name(name)
            # if experiment_id is not None:
            #    self.client.delete_experiment(experiment_id)
            raise

    def download_artifacts(self, base_save_path: Path, run: Run = None, runs: [] = None,
                           mlflow_folder: str = None) -> []:
        """
         Downloads all artifacts of the found runs. Creates download folder for each run
        @param base_save_path:  The path where the artifacts should be saved
        @param runs: Runs which should be considered
        @param run: The run which should be considered
        @param mlflow_folder: The specific folder to be downloaded for the given runs
        @return: Returns a list of created directories
        """

        if run is None and runs is None:
            raise ValueError("Please provide either a run to download or a list of runs")

        created_directories: list = []

        # Download single run
        if run is not None:
            run_save_path = Path(base_save_path, run.info.run_id)
            run_save_path = FolderManagement.create_directory(run_save_path, remove_if_exists=False)
            created_directories.append(run_save_path)
            if mlflow_folder is not None:
                self.client.download_artifacts(run_id=run.info.run_id, path=mlflow_folder,
                                               dst_path=str(run_save_path))
            else:
                self.client.download_artifacts(run_id=run.info.run_id, path="", dst_path=str(run_save_path))

            return created_directories

        # Download multiple runs
        for run in runs:
            try:
                run_path = Path(base_save_path, run.info.run_id)
                run_path = FolderManagement.create_directory(run_path, remove_if_exists=False)
                created_directories.append(run_path)
                if mlflow_folder is not None:
                    self.client.download_artifacts(run_id=run.info.run_id, path=mlflow_folder,
                                                   dst_path=str(run_path))
                else:
                    self.client.download_artifacts(run_id=run.info.run_id, path="", dst_path=str(run_path))

            except BaseException as ex:
                print(ex)
                continue

        return created_directories


