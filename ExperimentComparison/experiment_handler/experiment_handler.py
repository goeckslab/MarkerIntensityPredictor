import mlflow
from mlflow.entities import Run, RunInfo


class ExperimentHandler:
    client = mlflow.tracking.MlflowClient()
    # Cache for already downloaded runs
    __runs: list = []

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
    def get_runs(experiment_id: str) -> list:
        found_runs = []

        all_run_infos: [] = reversed(ExperimentHandler.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run: Run
            full_run = ExperimentHandler.client.get_run(run_info.run_id)

            # Skip unfinished or unsuccessful runs
            if full_run.info.status != 'FINISHED':
                continue

            if "Model" in full_run.data.tags:
                model = full_run.data.tags.get("Model")
                if model == "VAE" or model == "AE":
                    parent_run: Run = ExperimentHandler.__get_run_by_id(full_run.data.tags.get('mlflow.parentRunId'))
                    # Skip unfinished or unsuccessful runs
                    if parent_run is None or parent_run.info.status != "FINISHED":
                        continue

                    found_runs.append(full_run)

        return found_runs

    @staticmethod
    def __get_run_by_id(run_id: str) -> Run:
        run: Run

        # Check cache first
        for run in ExperimentHandler.__runs:
            if run.info.run_id == run_id:
                return run

        run: Run = ExperimentHandler.client.get_run(run_id)

        if run is not None:
            # Add to cache
            ExperimentHandler.__runs.append(run)

        return run
