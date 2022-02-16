import streamlit as st
import pandas as pd
import mlflow
from mlflow.entities import Experiment, Run
from experiment_handler.experiment_handler import ExperimentHandler
from data_management.data_management import DataManagement
from pathlib import Path

client = mlflow.tracking.MlflowClient()
temp_storage_folder = "latent_space_exploration_temp"

def check_session_state():
    """
    Initialize session state at startup
    @return:
    """
    if 'experiment' not in st.session_state:
        st.session_state.experiment = None

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = ""

    if "selected_experiment_id" not in st.session_state:
        st.session_state.selected_experiment_id = None

    if "selected_run_name" not in st.session_state:
        st.session_state.selected_run_name = None

    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None

    if "runs" not in st.session_state:
        st.session_state.runs = []


def load_experiments() -> list:
    return st.session_state.experiments


def disconnect():
    st.session_state.experiment = None
    st.session_state.experiments = None


def fetch_experiments(server_url: str):
    """
    Fetches experiments from the tracking server
    @param server_url:
    @return:
    """
    if tracking_server_url is not None:
        mlflow.set_tracking_uri = server_url

    st.session_state.experiments = client.list_experiments()


if __name__ == "__main__":
    check_session_state()
    experiments: list = load_experiments()
    if experiments is not None and len(experiments) != 0:
        st.sidebar.button("Disconnect", on_click=disconnect)

        exp_names = []
        exp: Experiment
        for exp in experiments:
            exp_names.append(exp.name)

        selected_experiment = st.sidebar.selectbox(
            'Select an experiment of your choice',
            exp_names, key="selected_experiment")

        for exp in experiments:
            if exp.name == st.session_state.selected_experiment:
                st.session_state.selected_experiment_id = exp.experiment_id

        if st.session_state.selected_experiment_id is not None:
            st.session_state.runs = ExperimentHandler.get_vae_runs(st.session_state.selected_experiment_id)
            run: Run
            run_names: list = []
            for run in st.session_state.runs:
                run_names.append(f"{run.data.tags.get('mlflow.runName')}")

            st.sidebar.selectbox("Select a run of your choice", run_names, key="selected_run_name")

        if st.session_state.selected_run_name is None:
            st.stop()

        run: Run
        for run in st.session_state.runs:
            if run.data.tags.get("mlflow.runName") == st.session_state.selected_run_name:
                st.session_state.selected_run = run

        if st.session_state.selected_run is None:
            st.write(f"Could not find a run matching the name: {st.session_state.selected_run_name}")
            st.stop()

        # shortcut
        selected_run = st.session_state.selected_run

        st.header(selected_run.data.tags.get('mlflow.runName'))
        st.sidebar.subheader("Run Information:")
        st.sidebar.write(f"File: {selected_run.data.params.get('file')}")
        st.sidebar.write(f"Epochs: {selected_run.data.params.get('epochs')}")
        st.sidebar.write(f"Input Dimensions: {selected_run.data.params.get('input_dimensions')}")
        st.sidebar.write(f"Latent Space: {selected_run.data.params.get('latent_space_dimension')}")
        st.sidebar.write(f"Morph: {selected_run.data.params.get('morphological_data')}")

        data_manager: DataManagement = DataManagement(Path(temp_storage_folder))
        # Download required artifacts
        data_manager.download_artifacts_for_run(st.session_state.selected_run.info.run_id)

        markers = pd.read_csv(Path(temp_storage_folder, "Base", "markers.csv"))['0'].to_list()
        with st.expander("Markers"):
            st.write(markers)

    else:
        st.title("Latent space exploration tool")
        st.write("Please enter a tracking url to connect to a mlflow server.")

        st.header("How to use?")
        tracking_server_url: str = st.sidebar.text_input("Tracking server url", value="",
                                                         help="Enter a valid tracking server url to connect. Leave blank for localhost connection")

        if tracking_server_url is None:
            st.sidebar.write("Please provide a valid url")

        st.sidebar.button("Connect", on_click=fetch_experiments, args=(tracking_server_url,))
