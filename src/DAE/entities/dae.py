from entities.data import Data
from services.data_loader import DataLoader
from services.args_parser import ArgumentParser
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from services.plots import Plots
import anndata as ad
import pandas as pd
from pathlib import Path
import umap
import tensorflow as tf
import sys


class DenoisingAutoEncoder:
    un_normalized_data: Data
    normalized_data: Data

    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    ae: any

    history: any

    input_dim: int
    encoding_dim: int

    input_umap: any
    latent_umap: any

    def __init__(self):
        self.inputs_dim = 0

    def load_data(self):
        print("Loading data...")
        self.un_normalized_data = Data()
        inputs, self.un_normalized_data.markers = DataLoader.get_data(ArgumentParser.get_args().file)

        self.un_normalized_data.inputs = np.array(inputs)

    def normalize(self, data):
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

    def split_data(self):
        print("Splitting data")
        X_dev, X_val = train_test_split(self.un_normalized_data.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

        self.un_normalized_data.X_train = X_train
        self.un_normalized_data.X_test = X_test
        self.un_normalized_data.X_val = X_val

        # Store the normalized data
        self.normalized_data = Data()
        self.normalized_data.markers = self.un_normalized_data.markers
        self.normalized_data.inputs = np.array(self.normalize(self.un_normalized_data.inputs))
        self.normalized_data.X_train = self.normalize(X_train)
        self.normalized_data.X_test = self.normalize(X_test)
        self.normalized_data.X_val = self.normalize(X_val)

        self.inputs_dim = self.normalized_data.inputs.shape[1]

    def add_noise(self):
        data = pd.DataFrame(self.normalized_data.X_train.copy())
        means = data.mean()
        std = data.std()

        for column in data:
            random_dist = np.random.normal(means[column], std[column], size=len(data[column]))
            r_column = pd.DataFrame(data[column].sample(frac=0.2))
            noise = np.random.choice(random_dist, size=len(r_column), replace=False)
            r_column[column] = noise

            for index in data.index:
                if index in r_column.index:
                    data[column][index] = r_column[column][index]

        self.normalized_data.X_train_noise = data.to_numpy()

    def build_auto_encoder(self):
        self.encoding_dim = 5
        activation = "linear"
        input_layer = keras.Input(shape=(self.inputs_dim,))

        # Encoder
        encoded = layers.Dense(20, activation=activation)(input_layer)
        encoded = layers.Dense(12, activation=activation)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation=activation)(encoded)

        # Decoder
        decoded = layers.Dense(12, activation=activation)(encoded)
        decoded = layers.Dense(20, activation=activation)(decoded)
        decoded = layers.Dense(self.inputs_dim, activation=activation)(decoded)

        # Auto encoder
        self.ae = keras.Model(input_layer, decoded)
        self.ae.summary()

        # Separate encoder model
        self.encoder = keras.Model(input_layer, encoded)
        self.encoder.summary()

        # Separate decoder model
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        deco = self.ae.layers[-3](encoded_input)
        deco = self.ae.layers[-2](deco)
        deco = self.ae.layers[-1](deco)
        # create the decoder model
        self.decoder = keras.Model(encoded_input, deco)
        self.decoder.summary()

        # Compile ae
        self.ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError(), metrics=['acc', 'mean_squared_error'])

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.history = self.ae.fit(self.normalized_data.X_train_noise, self.normalized_data.X_train_noise,
                                   epochs=500,
                                   batch_size=92,
                                   shuffle=True,
                                   callbacks=[callback],
                                   validation_data=(self.normalized_data.X_train, self.normalized_data.X_train))

    def predict(self):
        # Make some predictions
        cell = self.normalized_data.X_val[0]
        cell = cell.reshape(1, cell.shape[0])
        encoded_cell = self.encoder.predict(cell)
        decoded_cell = self.decoder.predict(encoded_cell)
        var_cell = self.ae.predict(cell)
        print(f"Epochs: {len(self.history.history['loss'])}")
        print(f"Input shape:\t{cell.shape}")
        print(f"Encoded shape:\t{encoded_cell.shape}")
        print(f"Decoded shape:\t{decoded_cell.shape}")
        print(f"\nInput:\n{cell[0]}")
        print(f"\nEncoded:\n{encoded_cell[0]}")
        print(f"\nDecoded:\n{decoded_cell[0]}")

    def create_h5ad_object(self):
        # Input
        fit = umap.UMAP()
        self.input_umap = input_umap = fit.fit_transform(self.normalized_data.X_train_noise)

        # latent space
        fit = umap.UMAP()
        encoded = self.encoder.predict(self.normalized_data.X_train)
        self.latent_umap = fit.fit_transform(encoded)

        self.__create_h5ad("latent_markers", self.latent_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        self.__create_h5ad("input", input_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        return

    def plots(self):
        Plots.plot_model_performance(self.history, f"model_performance_{self.encoding_dim}")
        Plots.plot_reconstructed_intensities(self.ae, self.normalized_data.X_val, self.normalized_data.markers,
                                             f"reconstructed_intensities_{self.encoding_dim}")
        Plots.latent_space_cluster(self.input_umap, self.latent_umap,
                                   f"latent_space_clusters_{self.encoding_dim}")

    def __create_h5ad(self, file_name: str, umap, markers, df):
        obs = pd.DataFrame(data=df, index=df.index)
        var = pd.DataFrame(index=markers)
        obsm = {"X_umap": umap}
        uns = dict()
        adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns, obsm=obsm)

        adata.write(Path(f'results/dae/{file_name}.h5ad'))
