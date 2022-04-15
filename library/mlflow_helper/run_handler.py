from mlflow.entities import Run, RunInfo
from typing import Optional


class RunHandler:
    client = None

    # Cache for already downloaded runs
    __runs: dict = {}

    def __init__(self, client):
        self.client = client

    def get_run_name_by_run_id(self, run_id: str, runs: []) -> Optional[str]:
        run: Run
        for run in runs:
            if run.info.run_id == run_id:
                return run.data.tags.get('mlflow.runName')

        return None

    def get_run_by_id(self, experiment_id: str, run_id: str) -> Optional[Run]:
        all_run_infos: [] = reversed(self.client.list_run_infos(experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self.client.get_run(run_info.run_id)

            if full_run.info.run_id == run_id:
                return full_run

        return None

    def get_summary_runs(self, experiment_id: str) -> list:
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

            if full_run.data.tags.get('mlflow.runName') == "Summary":
                parent_run: Run = self.__get_run_by_id(experiment_id=experiment_id,
                                                       run_id=full_run.data.tags.get('mlflow.parentRunId'))
                # Skip unfinished or unsuccessful runs
                if parent_run is None or parent_run.info.status != "FINISHED":
                    continue

                found_runs.append(full_run)

        return found_runs

    def get_inter_run_summary(self, experiment_id: str) -> Optional[Run]:
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
                    'mlflow.runName') == "Inter Run Summary":
                return full_run

        return None

    def get_model_run_id(self, args, model_experiment_id: str) -> str:
        model_run_id: str = self.get_run_id_by_name(experiment_id=model_experiment_id,
                                                    run_name=args.model[1])

        if model_run_id is None:
            raise ValueError(f"Could not find run with name {args.model[1]}")

        return model_run_id

    def get_run_id_by_name(self, experiment_id: str, run_name: str, parent_run_id: str = None) -> Optional[str]:
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
                if parent_run_id is not None and full_run.data.tags.get('mlflow.parentRunId') != parent_run_id:
                    continue

                # Add to cache
                if runs is None or len(runs) == 0:
                    self.__runs[experiment_id] = [full_run]
                else:
                    self.__runs[experiment_id].append(full_run)

                return full_run.info.run_id

        # Run not found
        return None

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
