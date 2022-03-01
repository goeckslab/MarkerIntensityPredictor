import streamlit as st
from pathlib import Path
from entities.data import Data


class SessionState:

    @staticmethod
    def reset_session_state():
        print("Reset session state...")
        st.session_state.experiment = None
        st.session_state.experiments = None
        st.session_state.runs = None
        st.session_state.selected_experiment_name = ""
        st.session_state.selected_experiment_id = None
        st.session_state.selected_run = None
        st.session_state.selected_run_name = None
        st.session_state.cells_to_generate = 0
        st.session_state.new_run_name = None
        st.session_state.new_run_completed = False
        st.session_state.tracking_server_url = "http://127.0.0.1:5000"
        st.session_state.connected = False
        st.session_state.data = Data()
        st.session_state.client = None
        # Path.unlink(Path("LatentSpaceExplorer", "tmp"))

    @staticmethod
    def initialize_session_state():
        """
        Initialize session state at startup
        @return:
        """

        print("Initialize session state")
        st.write("Initialize session state")

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

        if "tracking_server_url" not in st.session_state:
            st.session_state.tracking_server_url = "http://127.0.0.1:5000"

        if "connected" not in st.session_state:
            st.session_state.connected = False

        if "data" not in st.session_state:
            st.session_state.data = Data()

        if "dimension_to_fix" not in st.session_state:
            st.session_state.dimension_to_fix = None

        if "client" not in st.session_state:
            st.session_state.client = None
