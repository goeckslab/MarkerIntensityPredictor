import streamlit as st
import pandas as pd
from preprocessing.preprocessing import Preprocessing
from models.auto_encoder import AutoEncoder
from entities.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize(data):
    # Input data contains some zeros which results in NaN (or Inf)
    # values when their log10 is computed. NaN (or Inf) are problematic
    # values for downstream analysis. Therefore, zeros are replaced by
    # a small value; see the following thread for related discussion.
    # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

    data[data == 0] = 1e-32
    data = np.log10(data)

    standard_scaler = StandardScaler()
    data = standard_scaler.fit_transform(data)
    data = data.clip(min=-5, max=5)

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data = min_max_scaler.fit_transform(data)
    return data


st.title('Marker intensity prediction')
dataframe = pd.DataFrame()
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)

if not dataframe.empty:
    with st.expander("See raw data"):
        st.write(dataframe)

# Todo add retrain button
if not dataframe.empty:
    st.sidebar.slider("Test label")

if not dataframe.empty:
    inputs, markers = Preprocessing.get_data(dataframe)
    st.write(inputs)
    st.json(markers)
    data = Data(inputs=np.array(inputs), markers=markers, normalize=normalize)
    auto_encoder = AutoEncoder(data)
    auto_encoder.build_encoder()
    auto_encoder.build_decoder()
    auto_encoder.compile_auto_encoder()
    st.write(auto_encoder.encoder.summary())
    st.write(auto_encoder.decoder.summary())
    st.write(auto_encoder.auto_encoder.summary())
