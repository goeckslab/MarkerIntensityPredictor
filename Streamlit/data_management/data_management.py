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

    def __del(self):
        # Remove temp folder after use
        if self.__base_path.exists():
            Path.unlink(self.__base_path)

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

    def load_r2_scores_for_model(self, model: str) -> pd.DataFrame:
        """
        Loads all r2scores for the given model and combines them in a dataset
        @param model:
        @return:
        """
        combined_r2_scores = pd.DataFrame()
        markers = []

        path = Path(self.__base_path, "runs", model)
        for p in path.rglob("*"):
            if p.name == "r2_scores.csv":
                df = pd.read_csv(p.absolute(), header=0)

                # Get markers
                markers = df["Marker"].to_list()

                # Transponse
                df = df.T
                # Drop markers row
                df.drop(index=df.index[0],
                        axis=0,
                        inplace=True)

                combined_r2_scores = combined_r2_scores.append(df, ignore_index=True)

        combined_r2_scores.columns = markers

        return pd.DataFrame(columns=["Marker", "Score"],
                            data={"Marker": combined_r2_scores.columns, "Score": combined_r2_scores.mean().values})
