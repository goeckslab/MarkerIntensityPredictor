import streamlit as st
import pandas as pd
import mlflow
from mlflow.entities import Experiment, Run
from experiment_handler.experiment_handler import ExperimentHandler
from data_management.data_management import DataManagement
from pathlib import Path
from ui.ui import UIHandler
from laten_space_exploration import LatentSpaceExplorer

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

    if "selected_experiment_name" not in st.session_state:
        st.session_state.selected_experiment_name = ""

    if "selected_experiment_id" not in st.session_state:
        st.session_state.selected_experiment_id = None

    if "selected_run_name" not in st.session_state:
        st.session_state.selected_run_name = None

    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None

    if "runs" not in st.session_state:
        st.session_state.runs = []

    if "cells_to_generate" not in st.session_state:
        st.session_state.cells_to_generate = 0

    if "new_run_name" not in st.session_state:
        st.session_state.new_run_name = None

    if "new_run_completed" not in st.session_state:
        st.session_state.new_run_completed = False


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


def execute_latent_space():
    pass


if __name__ == "__main__":
    check_session_state()
    experiments: list = load_experiments()
    if experiments is not None and len(experiments) != 0:
        st.sidebar.button("Disconnect", on_click=disconnect)

        st.sidebar.selectbox(
            'Select an experiment of your choice',
            UIHandler.get_all_experiments_names(experiments), key="selected_experiment_name")

        for exp in experiments:
            if exp.name == st.session_state.selected_experiment_name:
                st.session_state.selected_experiment_id = exp.experiment_id

        if st.session_state.selected_experiment_id is None:
            st.stop()

        st.session_state.runs = ExperimentHandler.get_vae_runs(st.session_state.selected_experiment_id)
        st.sidebar.selectbox("Select a run of your choice", UIHandler.get_all_run_names(st.session_state.runs), index=0,
                             key="selected_run_name")

        if st.session_state.selected_run_name is None:
            st.stop()

        run: Run
        for run in st.session_state.runs:
            if run.data.tags.get("mlflow.runName") == st.session_state.selected_run_name:
                st.session_state.selected_run = run

        if st.session_state.selected_run is None:
            st.write(f"Could not find a run matching the name: {st.session_state.selected_run_name}")
            st.stop()

        # Show sidebar run information
        UIHandler.show_selected_run_information(st.session_state.selected_run)

        data_manager: DataManagement = DataManagement(Path(temp_storage_folder))
        # Download required artifacts
        data_manager.download_artifacts_for_run(st.session_state.selected_run.info.run_id)

        markers = pd.read_csv(Path(temp_storage_folder, "Base", "markers.csv"))['0'].to_list()
        with st.expander("Markers"):
            st.write(markers)

        embeddings = pd.read_csv(Path(temp_storage_folder, "VAE", "encoded_data.csv"))
        with st.expander("Embeddings"):
            st.write(embeddings)

        # Latent space exploration
        st.subheader("Latent space")

        cell_number_col, run_name_col, fixed_dimension_col = st.columns(3)
        cell_number_col.number_input("How many cells should be generated?", min_value=100, key="cells_to_generate")

        # Stop execution if one pass fails
        stop_execution: bool = False

        if st.session_state.cells_to_generate == 0:
            cell_number_col.write("Please provide value greater 100")
            stop_execution = True

        run_name_col.text_input("Please provide a name for the new run", key="new_run_name")

        if st.session_state.new_run_name is None:
            run_name_col.write("Please provide a valid run name.")
            st.session_state.new_run_completed = False
            stop_execution = True

        if ExperimentHandler.run_exists(st.session_state.selected_experiment_id, st.session_state.new_run_name):
            st.write("This run does already exist. Please specify a different run name")
            stop_execution = True

        if stop_execution:
            st.stop()

        if st.button("Generate cells", disabled=st.session_state.new_run_completed,
                     on_click=execute_latent_space):  # TODO: Add on click handler
            with mlflow.start_run(experiment_id=st.session_state.selected_experiment_id,
                                  run_name=st.session_state.new_run_name) as new_run:
                mlflow.log_param("cells_to_generate", st.session_state.cells_to_generate)

                latent_space_explorer: LatentSpaceExplorer = LatentSpaceExplorer(embeddings=embeddings, markers=markers,
                                                                                 base_results_path=Path(
                                                                                     temp_storage_folder))
                latent_space_explorer.generate_new_cells(st.session_state.cells_to_generate)
                latent_space_explorer.plot_generated_cells()
                latent_space_explorer.plot_generated_cells_differences()
                latent_space_explorer.umap_mapping_of_generated_cells()

                st.write("Latent space exploration done.")
                # Needs to be always at last position
                st.session_state.new_run_completed = True










    else:
        st.title("Latent space exploration tool")
        st.write("Please enter a tracking url to connect to a mlflow server.")

        st.header("How to use?")
        tracking_server_url: str = st.sidebar.text_input("Tracking server url", value="",
                                                         help="Enter a valid tracking server url to connect. Leave blank for localhost connection")

        if tracking_server_url is None:
            st.sidebar.write("Please provide a valid url")

        st.sidebar.button("Connect", on_click=fetch_experiments, args=(tracking_server_url,))
