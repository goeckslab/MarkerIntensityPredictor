import mlflow


class ExperimentHandler:
    client = mlflow.tracking.MlflowClient()

    @staticmethod
    def get_experiment_id_by_name(experiment_name: str, create_experiment: bool = True) -> str:
        """
        Gets the experiment id associated with the given experiment name.
        If no experiment is found by default a new experiment will be created
        @param experiment_name:
        @param create_experiment:
        @return: The experiment id
        """
        found_experiment_id = 0

        experiments = ExperimentHandler.client.list_experiments()  # returns a list of mlflow.entities.Experiment
        for experiment in experiments:
            if experiment.name == experiment_name:
                found_experiment_id = experiment.experiment_id

        if found_experiment_id == 0 and create_experiment:
            found_experiment_id = ExperimentHandler.create_experiment(name=experiment_name)

        return found_experiment_id

    @staticmethod
    def create_experiment(name: str) -> str:
        """
        Creates a new experiment with the given name
        @param name:
        @return: The string of the newly created experiment
        """
        try:
            return ExperimentHandler.client.create_experiment(name=name)
        except BaseException as ex:
            raise
