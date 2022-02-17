import streamlit as st


class SessionState:

    @staticmethod
    def reset_session_state():
        st.session_state.experiment = None
        st.session_state.experiments = None
        st.session_state.runs = None
        st.selected_experiment_name = ""
        st.session_state.selected_experiment_id = None
        st.session_state.selected_run = None
        st.session_state.selected_run_name = None
        st.session_state.cells_to_generate = 0
        st.session_state.new_run_name = None
        st.session_state.new_run_completed = False
        st.session_state.tracking_server_url = ""
        st.session_state.connected = False

    @staticmethod
    def initialize_session_state():
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

        if "tracking_server_url" not in st.session_state:
            st.session_state.tracking_server_url = ""

        if "connected" not in st.session_state:
            st.session_state.connected = False
