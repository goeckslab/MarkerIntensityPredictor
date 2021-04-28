import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vae import VAE, Sampling
from plots import Plots
from args_parser import ArgumentParser

class Data:
    inputs: pd.DataFrame()
    markers: pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()


class CellStateVAE:
    # The latent space dimensionality
    latent_dim: int
    # The defined encoder
    encoder = None

    # The defined decoder
    decoder = None
    # The vae
    vae = None

    un_normalized_data: Data
    normalized_data: Data

    markers = None
    # inputs = pd.DataFrame()
    inputs_dim: int
    history = None

    # The neuron count before the latent space
    neurons_before_latent: np.ndarray

    def __init__(self):
        self.latent_dim = 6

    def load_data(self):
        print("Loading data...")
        self.un_normalized_data = Data()
        self.un_normalized_data.inputs, self.un_normalized_data.markers = DataLoader.get_data(
            ArgumentParser.get_args().file)

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
        data = data.clip(min=-10, max=10)

        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        data = min_max_scaler.fit_transform(data)
        return data

    def split_data(self):
        print("Splitting data")
        X_dev, X_val = train_test_split(self.un_normalized_data.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

        self.inputs_dim = self.un_normalized_data.inputs.shape[1]

        # This is primarily for the purpose of
        # being able to see how much data is removed
        # (as part of outlier removal) and plot
        # the changes in data distribution.

        self.un_normalized_data.X_train = X_train
        self.un_normalized_data.X_test = X_test
        self.un_normalized_data.X_val = X_val

        self.normalized_data = Data()
        self.normalized_data.markers = self.un_normalized_data.markers
        self.normalized_data.inputs = self.normalize(self.un_normalized_data.inputs)
        self.normalized_data.X_train = self.normalize(X_train)
        self.normalized_data.X_test = self.normalize(X_test)
        self.normalized_data.X_val = self.normalize(X_val)

    def build_encoder(self):
        print("Building encoder...")
        r = regularizers.l1(10e-5)
        activation = tf.nn.relu

        encoder_inputs = keras.Input(shape=self.inputs_dim)
        h1 = layers.Dense(self.inputs_dim, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(self.inputs_dim / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(self.inputs_dim / 3, activation=activation, activity_regularizer=r)(h2)

        # neurons count before latent dim
        self.neurons_before_latent = np.prod(keras.backend.int_shape(h3)[1:])

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(h3)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(h3)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

    def build_decoder(self):
        print("Building decoder...")
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        h1 = layers.Dense(self.neurons_before_latent, activation=tf.nn.relu)(decoder_inputs)
        h2 = layers.Dense(self.inputs_dim / 2, activation=tf.nn.relu)(h1)

        decoder_outputs = layers.Dense(self.inputs_dim)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    def train(self):
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=keras.optimizers.Adam(lr=0.0005))
        self.history = self.vae.fit(self.normalized_data.X_train,
                                    validation_data=(self.normalized_data.X_test, self.normalized_data.X_test),
                                    epochs=100,
                                    batch_size=32, shuffle=True, verbose=0)

    def predict(self):
        # Make some predictions
        cell = self.normalized_data.X_val[0]
        cell = cell.reshape(1, cell.shape[0])
        mean, log_var, z = self.encoder.predict(cell)
        encoded_cell = z
        decoded_cell = self.decoder.predict(encoded_cell)
        var_cell = self.vae.predict(cell)
        print(f"Input shape:\t{cell.shape}")
        print(f"Encoded shape:\t{encoded_cell.shape}")
        print(f"Decoded shape:\t{decoded_cell.shape}")
        print(f"\nInput:\n{cell[0]}")
        print(f"\nEncoded:\n{encoded_cell[0]}")
        print(f"\nDecoded:\n{decoded_cell[0]}")

    def create_plots(self):
        Plots.plot_distribution_of_latent_variables(self.encoder, self.normalized_data.X_train, self.latent_dim, step_size, z)
        Plots.plot_model_performance(self.history)
        Plots.plot_markers(X_train, X_test, X_val, markers)
        Plots.plot_reconstructed_markers(z_grid, x_pred_grid, markers)
        Plots.latent_space_cluster(X_train, vae)
        Plots.plot_reconstructed_intensities(vae, X_val, markers)

    def plot_label_clusters(self, vae, data, labels):
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = vae.encoder.predict(data)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

    # (x_train, y_train), _ = keras.datasets.mnist.load_data()
    # x_train = np.expand_dims(x_train, -1).astype("float32") / 255

    # plot_label_clusters(self.vae, x_train, y_train)


if __name__ == "__main__":
    vae = CellStateVAE()

    vae.load_data()
    vae.split_data()
    vae.build_encoder()
    vae.build_decoder()
