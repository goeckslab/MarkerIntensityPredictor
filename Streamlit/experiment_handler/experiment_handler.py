import mlflow


class ExperimentHandler:
    client = mlflow.tracking.MlflowClient()

    @staticmethod
    def get_experiment_id_by_name(experiment_name: str) -> str:
        """
        Gets the experiment id associated with the given experiment name.
        If no experiment is found by default a new experiment will be created
        @param experiment_name:
        @return: The experiment id
        """
        found_experiment_id = None

        experiments = ExperimentHandler.client.list_experiments()  # returns a list of mlflow.entities.Experiment
        for experiment in experiments:
            if experiment.name == experiment_name:
                found_experiment_id = experiment.experiment_id

        return found_experiment_id

    @staticmethod
    def get_vae_runs(experiment_id: str) -> list:
        """
        Returns all runs associated to a vae
        @param experiment_id:
        @return:
        """
        found_runs = []

        all_run_infos = reversed(ExperimentHandler.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run = ExperimentHandler.client.get_run(run_info.run_id)
            if "Model" in full_run.data.tags:
                model = full_run.data.tags.get("Model")
                if model == "VAE":
                    found_runs.append(full_run)

        return found_runs

    @staticmethod
    def run_exists(experiment_id: str, run_name: str) -> bool:
        all_run_infos = reversed(ExperimentHandler.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run = ExperimentHandler.client.get_run(run_info.run_id)
            if full_run.data.tags.get('mlflow.runName') == run_name:
                return True

        return False
