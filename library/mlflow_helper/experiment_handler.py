import mlflow.exceptions
from mlflow.entities import Run, RunInfo
from pathlib import Path
from typing import Optional
from library.data.folder_management import FolderManagement
from mlflow.exceptions import ErrorCode


class ExperimentHandler:
    # Cache for already downloaded runs
    __runs: dict = {}

    def __init__(self, client):
        self._client = client

    @property
    def client(self):
        return self._client

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

        experiments = self._client.list_experiments()  # returns a list of mlflow.entities.Experiment
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
            experiment_id: str = self._client.create_experiment(name=name)
            self._client.set_experiment_tag(experiment_id, "description", description)
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
