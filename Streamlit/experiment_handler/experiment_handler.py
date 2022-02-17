import mlflow
import streamlit as st
from mlflow import exceptions


class ExperimentHandler:
    client = None

    def __init__(self):
        self.client = mlflow.tracking.MlflowClient(tracking_uri=st.session_state.tracking_server_url)

    def fetch_experiments(self) -> list:
        """
        Fetches experiments from the tracking server
        @return:  List of experiments downloaded from the tracking server
        """
        try:
            st.session_state.experiments = self.client.list_experiments()
            return st.session_state.experiments
        except exceptions.MlflowException as mlex:
            st.write(mlex)
            return None

        except BaseException as ex:
            return None

    def get_experiment_id_by_name(self, experiment_name: str) -> str:
        """
        Gets the experiment id associated with the given experiment name.
        If no experiment is found by default a new experiment will be created
        @param experiment_name:
        @return: The experiment id
        """
        found_experiment_id = None

        experiments = self.client.list_experiments()  # returns a list of mlflow.entities.Experiment
        for experiment in experiments:
            if experiment.name == experiment_name:
                found_experiment_id = experiment.experiment_id

        return found_experiment_id

    def get_vae_runs(self, experiment_id: str) -> list:
        """
        Returns all runs associated to a vae
        @param experiment_id:
        @return:
        """
        found_runs = []

        all_run_infos = reversed(self.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run = self.client.get_run(run_info.run_id)
            if "Model" in full_run.data.tags:
                model = full_run.data.tags.get("Model")
                if model == "VAE":
                    found_runs.append(full_run)

        return found_runs

    def run_exists(self, experiment_id: str, run_name: str) -> bool:
        all_run_infos = reversed(self.client.list_run_infos(experiment_id))
        for run_info in all_run_infos:
            full_run = self.client.get_run(run_info.run_id)
            if full_run.data.tags.get('mlflow.runName') == run_name:
                return True

        return False