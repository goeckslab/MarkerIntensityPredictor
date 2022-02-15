import streamlit as st
import pandas as pd
import mlflow
from mlflow.entities import Experiment, Run
from experiment_handler.experiment_handler import ExperimentHandler

st.title('VAE latent space explorer')
dataframe = pd.DataFrame()
client = mlflow.tracking.MlflowClient()


def check_session_state():
    if 'experiment' not in st.session_state:
        st.session_state.experiment = None

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = ""


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

        if st.session_state.selected_experiment != "":
            runs = ExperimentHandler.get_runs()
            run: Run
            run_names: list = []
            for run in runs:
                st.write(run)
                # run_names.append(run.data)



    else:
        tracking_server_url: str = st.sidebar.text_input("Tracking server url", value="")
        if tracking_server_url is None:
            st.sidebar.write("Please provide a valid url")

        st.sidebar.button("Connect", on_click=fetch_experiments, args=(tracking_server_url,))
