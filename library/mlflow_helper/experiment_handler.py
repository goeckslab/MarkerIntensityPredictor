from mlflow.entities import Run
from pathlib import Path
import pandas as pd
from typing import Tuple
from library.data.folder_management import FolderManagement


class ExperimentHandler:
    client = None

    # Cache for already downloaded runs
    __runs: list = []

    def __init__(self, client):
        self.client = client

    def get_experiment_id_by_name(self, experiment_name: str, experiment_description: str,
                                  create_experiment: bool = True) -> str:
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

    def create_experiment(self, name: str, description: str) -> str:
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
        except BaseException as ex:
            # print("Experiment does already exists. Deleting and recreating experiment.")
            # experiment_id = self.client.get_experiment_by_name(name)
            # if experiment_id is not None:
            #    self.client.delete_experiment(experiment_id)
            raise

    def get_runs(self, experiment_id: str) -> list:
        found_runs = []

        all_run_infos: [] = reversed(self.client.list_run_infos(experiment_id))

        for run_info in all_run_infos:
            full_run: Run
            full_run = self.client.get_run(run_info.run_id)

            # Skip unfinished or unsuccessful runs
            if full_run.info.status != 'FINISHED':
                continue

            if "Model" in full_run.data.tags:
                model = full_run.data.tags.get("Model")
                if model == "VAE" or model == "AE":
                    parent_run: Run = self.__get_run_by_id(full_run.data.tags.get('mlflow.parentRunId'))
                    # Skip unfinished or unsuccessful runs
                    if parent_run is None or parent_run.info.status != "FINISHED":
                        continue

                    found_runs.append(full_run)

        return found_runs

    def download_artifacts(self, save_path: Path, runs: [], mlflow_folder: str):
        """
         Downloads all artifacts of the found experiments
        @param save_path:  The path where the artificats should be saved
        @param runs: Runs which should be considered
        @param mlflow_folder: The specific folder to be downloaded for the given runs
        @return:
        """
        print("Downloading artifacts...")
        for run in runs:

            try:
                run_path = Path(save_path, run.info.run_id)
                run_path = FolderManagement.create_directory(run_path, remove_if_exists=False)
                self.client.download_artifacts(run.info.run_id, mlflow_folder,
                                               str(run_path))

            except BaseException as ex:
                print(ex)
                continue

        print("Downloading finished.")


    def __get_run_by_id(self, run_id: str) -> Run:
        run: Run

        # Check cache first
        for run in self.__runs:
            if run.info.run_id == run_id:
                return run

        run: Run = self.client.get_run(run_id)

        if run is not None:
            # Add to cache
            self.__runs.append(run)

        return run
