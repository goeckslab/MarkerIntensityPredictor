import pandas as pd
import streamlit as st
from pathlib import Path
import mlflow


class DataManagement:
    client = None
    __base_path: Path

    def __init__(self, root_path: Path):
        self.client = mlflow.tracking.MlflowClient(tracking_uri=st.session_state.tracking_server_url)
        self.__base_path = Path(root_path)
        # Create path if it does not exist
        if not self.__base_path.exists():
            Path.mkdir(self.__base_path)

    def download_artifacts_for_run(self, run_id: str):
        """
        Downloads all artifacts of the found experiments
        @return:
        """
        # Create temp directory
        try:
            self.client.download_artifacts(run_id, "base", str(self.__base_path))
            self.client.download_artifacts(run_id, "VAE", str(self.__base_path))
        except BaseException as ex:
            print(ex)
