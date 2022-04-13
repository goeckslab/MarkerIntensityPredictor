from mlflow.entities import Run, RunInfo
from typing import Optional


class RunHandler:
    client = None

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
