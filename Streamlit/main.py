import streamlit as st
import pandas as pd
from preprocessing.preprocessing import Preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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

    # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # data = min_max_scaler.fit_transform(data)
    return data


st.title('Marker intensity prediction')
dataframe = pd.DataFrame()
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe =  pd.read_csv(uploaded_file)

if not dataframe.empty:
    with st.expander("See raw data"):
        st.write(dataframe)

if not dataframe.empty:
    inputs, markers = Preprocessing.get_data(dataframe)
    with st.expander("Working data:"):
        st.header("Cleaned inputs:")
        st.write(inputs)
        st.header("Markers used for training:")
        st.write(markers)

    data = Data(inputs=np.array(inputs), markers=markers, normalize=normalize)
    vae = VariationalAutoEncoder(data)
    with st.spinner('Training model...'):
        vae.build_auto_encoder()

    train_history = pd.DataFrame()
    train_history['kl_loss'] = vae.history.history['kl_loss']
    train_history['loss'] = vae.history.history['loss']
    train_history['reconstruction_loss'] = vae.history.history['reconstruction_loss']
    st.line_chart(train_history)

    # Test prediction
    cell, latent_cell, decoded_cell = vae.predict()
    reconstructed_x, x = vae.reconstruction()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
    sns.heatmap(x, ax=ax1, xticklabels=markers)
    sns.heatmap(reconstructed_x, ax=ax2, xticklabels=markers)

    ax1.set_title("X Test")
    ax2.set_title("Reconstructed X Test")
    fig.tight_layout()
    st.pyplot(fig)
    st.write(r2_score(x, reconstructed_x))
    st.write(pd.DataFrame(x).head(10))
    st.write(pd.DataFrame(reconstructed_x).head(10))
    # st.write(vae.vae.evaluate())
