import mlflow
from mlflow.exceptions import RESOURCE_ALREADY_EXISTS


class ExperimentHandler:
    client = None

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

        if found_experiment_id == 0 and create_experiment:
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
            return self.client.create_experiment(name=name)
        except RESOURCE_ALREADY_EXISTS as ex:
            print("Experiment does already exists. Deleting and recreating experiment.")
            experiment_id = self.client.get_experiment_by_name(name)
            if experiment_id is not None:
                self.client.delete_experiment(experiment_id)
        except BaseException as ex:
            raise
