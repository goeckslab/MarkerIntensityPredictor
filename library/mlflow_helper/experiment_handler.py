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

    def get_vae_ae_runs(self, experiment_id: str) -> list:
        """
        Returns all runs created by a vae or ae
        @param experiment_id:
        @return:
        """
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
                if model == "VAE" or model == "AE" or model == "ElasticNet":
                    parent_run: Run = self.__get_run_by_id(experiment_id=experiment_id,
                                                           run_id=full_run.data.tags.get('mlflow.parentRunId'))
                    # Skip unfinished or unsuccessful runs
                    if parent_run is None or parent_run.info.status != "FINISHED":
                        continue

                    found_runs.append(full_run)

        return found_runs

    def get_run_comparison_run(self, experiment_id: str) -> Optional[Run]:
        """
        Returns the most recent run comparison
        @param experiment_id:
        @return: A run or None
        """
        all_run_infos: [] = reversed(self.client.list_run_infos(experiment_id))
        end_time: int = 0
        for run_info in all_run_infos:
            full_run: Run
            full_run = self.client.get_run(run_info.run_id)

            # Skip unfinished or unsuccessful runs
            if full_run.info.status != 'FINISHED':
                continue

            if full_run.info.end_time > end_time and full_run.info.status == "FINISHED" and full_run.data.tags.get(
                    'mlflow.runName') == "Run Comparison":
                return full_run

        return None

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
                self.client.download_artifacts(run.info.run_id, mlflow_folder,
                                               str(run_save_path))
            else:
                self.client.download_artifacts(run.info.run_id, path="", dst_path=str(run_save_path))

            return created_directories

        # Download multiple runs
        for run in runs:
            try:
                run_path = Path(base_save_path, run.info.run_id)
                run_path = FolderManagement.create_directory(run_path, remove_if_exists=False)
                created_directories.append(run_path)
                if mlflow_folder is not None:
                    self.client.download_artifacts(run.info.run_id, mlflow_folder,
                                                   str(run_path))
                else:
                    self.client.download_artifacts(run.info.run_id, str(run_path))

            except BaseException as ex:
                print(ex)
                continue

        return created_directories

    def get_run_by_name(self, experiment_id: str, run_name: str) -> Optional[Run]:
        """
        Returns a run for a given name in a given experiment
        @param experiment_id: The experiment id in which the run is located
        @param run_name:  The run name to search for
        @return: A run or None if not found
        """
        run: Run

        # Check cache
        runs: [] = self.__runs.get(experiment_id)

        if runs is not None and len(runs) != 0:
            for run in runs:
                if run.data.tags.get('mlflow.runName') == run_name:
                    return run

        # Run not cached
        all_run_infos: [] = reversed(self.client.list_run_infos(experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self.client.get_run(run_info.run_id)

            if full_run.data.tags.get('mlflow.runName') == run_name:
                # Add to cache
                if runs is None or len(runs) == 0:
                    self.__runs[experiment_id] = [full_run]
                else:
                    self.__runs[experiment_id].append(full_run)

                return full_run

        # Run not found
        return None

    def get_run_id_by_name(self, experiment_id: str, run_name: str) -> Optional[str]:
        """
        Returns a run id for a given name in a given experiment
        @param experiment_id: The experiment id in which the run is located
        @param run_name:  The run name to search for
        @return: A run or None if not found
        """
        run: Run

        # Check cache
        runs: [] = self.__runs.get(experiment_id)

        if runs is not None and len(runs) != 0:
            for run in runs:
                if run.data.tags.get('mlflow.runName') == run_name:
                    return run.info.run_id

        # Run not cached
        all_run_infos: [] = reversed(self.client.list_run_infos(experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self.client.get_run(run_info.run_id)

            if full_run.data.tags.get('mlflow.runName') == run_name:
                # Add to cache
                if runs is None or len(runs) == 0:
                    self.__runs[experiment_id] = [full_run]
                else:
                    self.__runs[experiment_id].append(full_run)

                return full_run.info.run_id

        # Run not found
        return None

    def __get_run_by_id(self, experiment_id: str, run_id: str) -> Run:
        run: Run

        # Check cache first

        runs: [] = self.__runs.get(experiment_id)

        if runs is not None and len(runs) != 0:
            for run in runs:
                if run.info.run_id == run_id:
                    return run

        run: Run = self.client.get_run(run_id)

        if run is not None:
            # Add to cache
            if runs is None or len(runs) == 0:
                self.__runs[experiment_id] = [run]
            else:
                self.__runs[experiment_id].append(run)

        return run
