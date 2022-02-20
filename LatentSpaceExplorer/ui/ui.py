import streamlit as st
from mlflow.entities import Run, Experiment


class UIHandler:
    @staticmethod
    def show_selected_run_information(selected_run):
        st.header(selected_run.data.tags.get('mlflow.runName'))
        st.sidebar.subheader("Run Information:")
        st.sidebar.write(f"File: {selected_run.data.params.get('file')}")
        st.sidebar.write(f"Epochs: {selected_run.data.params.get('epochs')}")
        st.sidebar.write(f"Input Dimensions: {selected_run.data.params.get('input_dimensions')}")
        st.sidebar.write(f"Latent Space: {selected_run.data.params.get('latent_space_dimension')}")
        st.sidebar.write(f"Morph: {selected_run.data.params.get('morphological_data')}")

    @staticmethod
    def get_all_run_names(runs: dict) -> list:
        run: Run
        run_names: list = [None]
        for key, value in runs.items():
            run_names.append(key)

        return run_names

    @staticmethod
    def get_all_experiments_names(experiments: list) -> list:
        exp_names = []
        exp: Experiment
        for exp in experiments:
            exp_names.append(exp.name)

        return exp_names
