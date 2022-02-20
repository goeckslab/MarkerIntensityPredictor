import streamlit as st
import pandas as pd
import mlflow
from mlflow.entities import Experiment, Run
from experiment_handler.experiment_handler import ExperimentHandler
from data_management.data_management import DataManagement
from pathlib import Path
from ui.ui import UIHandler
from laten_space_exploration import LatentSpaceExplorer
from sessions.session_state import SessionState

temp_storage_folder = Path("LatentSpaceExplorer", "tmp")


def ml_client():
    return mlflow.tracking.MlflowClient(tracking_uri=st.session_state.tracking_server_url)


def load_experiments() -> list:
    return st.session_state.experiments


def connected():
    st.session_state.connected = True


def run_name_changed():
    st.session_state.new_run_completed = False


def execute_latent_space_exploration():
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
        st.session_state.new_run_completed = True


if __name__ == "__main__":
    SessionState.initialize_session_state()

    if st.session_state.connected:
        experiment_handler = ExperimentHandler()
        experiments: list = experiment_handler.fetch_experiments()
        st.sidebar.button("Disconnect", on_click=SessionState.reset_session_state)
        st.write(st.session_state.tracking_server_url)

        st.sidebar.selectbox(
            'Select an experiment of your choice',
            UIHandler.get_all_experiments_names(experiments), key="selected_experiment_name")

        for exp in experiments:
            if exp.name == st.session_state.selected_experiment_name:
                st.session_state.selected_experiment_id = exp.experiment_id

        if st.session_state.selected_experiment_id is None:
            st.stop()

        st.session_state.runs = experiment_handler.get_vae_runs(st.session_state.selected_experiment_id)
        st.sidebar.selectbox("Select a run of your choice", UIHandler.get_all_run_names(st.session_state.runs), index=0,
                             key="selected_run_name")

        if st.session_state.selected_run_name is None:
            st.stop()

        run: Run
        for parent_run, vae_run in st.session_state.runs.items():
            if parent_run == st.session_state.selected_run_name:
                st.session_state.selected_run = vae_run

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

        run_name_col.text_input("Please provide a name for the new run", key="new_run_name", on_change=run_name_changed)

        if not st.session_state.new_run_completed:
            if st.session_state.new_run_name is None:
                run_name_col.write("Please provide a valid run name.")
                stop_execution = True

            if experiment_handler.run_exists(st.session_state.selected_experiment_id, st.session_state.new_run_name):
                st.write("This run does already exist. Please specify a different run name")
                stop_execution = True

            if stop_execution:
                st.stop()

            st.button("Generate cells", disabled=st.session_state.new_run_completed,
                      on_click=execute_latent_space_exploration)
            st.stop()

        st.header("Results")
        generated_cells: pd.DataFrame = pd.read_csv(Path(temp_storage_folder, "generated", "generated_cells.csv"))
        with st.expander("Encoded data"):
            st.write(generated_cells)

        





    else:
        st.title("Latent space exploration tool")
        st.write("Please enter a tracking url to connect to a mlflow server.")

        st.header("How to use?")

        st.sidebar.text_input("Tracking server url", value="", placeholder="http://127.0.0.1:5000",
                              help="Enter a valid tracking server url to connect. Leave blank for localhost connection",
                              key="tracking_server_url")

        if st.session_state.tracking_server_url == "":
            st.sidebar.write("Please provide a valid url")
            st.stop()

        # experiment_handler: ExperimentHandler = ExperimentHandler()
        st.sidebar.button("Connect", on_click=connected)
