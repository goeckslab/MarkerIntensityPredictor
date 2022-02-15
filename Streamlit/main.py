import streamlit as st
import pandas as pd
import mlflow
from experiment import Experiment
from time import sleep
import numpy as np

st.title('VAE latent space explorer')
dataframe = pd.DataFrame()


def check_session_state():
    if 'experiment' not in st.session_state:
        st.session_state.experiment = None


def load_experiment() -> Experiment:
    return st.session_state.experiment


def create_experiment(experiment_name: str):
    st.session_state.experiment = Experiment(experiment_name=experiment_name)


def clear_experiment():
    st.session_state.experiment = None


if __name__ == "__main__":
    check_session_state()
    experiment: Experiment = load_experiment()
    if experiment is not None:
        st.header(f"{experiment.name}")

        st.sidebar.button("Finish experiment", on_click=clear_experiment)


    else:
        name: str = st.sidebar.text_input("Experiment Name")
        if name is None:
            st.sidebar.write("Please provide a valid name")

        st.sidebar.button("Search", on_click=create_experiment, args=(name,))
