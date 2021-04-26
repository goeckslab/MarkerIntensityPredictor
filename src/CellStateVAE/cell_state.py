import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from file_service import FileService
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vae import VAE, Sampling

class Data:
    inputs: pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()
    y_train: pd.DataFrame()
    y_test: pd.DataFrame()
    y_val: pd.DataFrame()

    def __init__(self, inputs, X_train, X_test, X_val, y_train, y_test, y_val):
        self.inputs = inputs
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val



class CellStateVAE:
    # The latent space dimensionality
    latent_dim: int
    # The defined encoder
    encoder = None

    # The defined decoder
    decoder = None
    # The vae
    vae = None

    unnormalized_data: Data

    markers = None
    inputs = pd.DataFrame()
    inputs_dim: int

    # The neuron count before the latent space
    neurons_before_latent: np.ndarray

    def __init__(self):
        self.latent_dim = 6

    def load_data(self):
        print("Loading data...")
        self.inputs, self.markers = FileService.get_data("./data/HTA9-2_Bx1_HMS_Tumor_quant.csv")

    def prepare_data(self):
        print("preparing helper variables...")
        X_dev, X_val = train_test_split(self.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)



        self.inputs_dim = self.inputs.shape[1]

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
        print("splitting data")
        X_dev, X_val = train_test_split(self.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

        # This is primarily for the purpose of
        # being able to see how much data is removed
        # (as part of outlier removal) and plot
        # the changes in data distribution.

        init_inputs = self.inputs
        init_X_train = X_train
        init_X_test = X_test
        init_X_val = X_val

        self.inputs = self.normalize(self.inputs)
        self.X_train = self.normalize(X_train)
        self.X_test = self.normalize(X_test)
        self.X_val = self.normalize(X_val)

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
        print("building decoder...")
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        h1 = layers.Dense(self.neurons_before_latent, activation=tf.nn.relu)(decoder_inputs)
        h2 = layers.Dense(self.inputs_dim / 2, activation=tf.nn.relu)(h1)

        decoder_outputs = layers.Dense(self.inputs_dim)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    def train(self):
        vae = VAE(self.encoder, self.decoder)
        vae.compile(optimizer=keras.optimizers.Adam(lr=0.0005))
        history = vae.fit(X_train, validation_data=(X_test, X_test), epochs=100, batch_size=32, shuffle=True, verbose=0)

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
    vae.prepare_helper_vars()
    vae.build_encoder()
    vae.build_decoder()
