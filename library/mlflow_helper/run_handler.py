from mlflow.entities import Run
from typing import Optional


class RunHandler:

    @staticmethod
    def get_run_name_by_run_id(run_id: str, runs: []) -> Optional[str]:
        run: Run
        for run in runs:
            if run.info.run_id == run_id:
                return run.data.tags.get('mlflow.runName')

        return None
