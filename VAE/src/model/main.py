from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import randrange
import pandas as pd
from pathlib import Path
import keras
from src.model.sampling import Sampling
from src.model.vae_model import VAE
import anndata as ad
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score
import umap
import tensorflow as tf
from src.data.data import Data
import mlflow
import mlflow.tensorflow


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class VAutoEncoder:
    # The split data
    data: Data
    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    vae: any

    # the training history of the AE
    history: any

    # the latent space dimensions
    latent_space_dimensions: int

    r2_scores = pd.DataFrame(columns=["Marker", "Score"])
    rmse = pd.DataFrame()

    # Latent space encoded data
    encoded_data = pd.DataFrame()

    # Reconstructed data from the latent space
    reconstructed_data = pd.DataFrame()

    args = None
    # The default results folder, where all experiment data is being stored. Can be overriden by the constructor
    results_folder = Path("results")

    def __init__(self, args, cells: pd.DataFrame, markers: list, latent_space_dimensions=10, activation="relu"):
        self.latent_space_dimensions = latent_space_dimensions
        self.args = args
        self.data = Data(inputs=np.array(cells), markers=markers, normalize=self.normalize)
        self.activation = activation

        mlflow.log_param("input_dimensions", self.data.inputs_dim)
        mlflow.log_param("activation", self.activation)

    @staticmethod
    def normalize(inputs: np.ndarray):
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        inputs[inputs == 0] = 1e-32
        inputs = np.log10(inputs)

        standard_scaler = StandardScaler()
        inputs = standard_scaler.fit_transform(inputs)
        inputs = inputs.clip(min=-5, max=5)

        # min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        # inputs = min_max_scaler.fit_transform(inputs)

        return inputs

    def check_mean_and_std(self):
        rnd = randrange(0, self.data.inputs.shape[1])
        # Mean should be zero and standard deviation
        # should be 1. However, due to some challenges
        # relationg to floating point positions and rounding,
        # the values should be very close to these numbers.
        # For details, see:
        # https://stackoverflow.com/a/40405912/947889
        # Hence, we assert the rounded values.
        print(self.data.inputs)
        print(self.data.inputs[:, rnd])

        print(f"Std: {self.data.inputs[:, rnd].std()}")
        print(f"Mean: {self.data.inputs[:, rnd].mean()}")

    def build_auto_encoder(self):
        """

        """

        mlflow.keras.autolog()

        inputs_dim = self.data.inputs_dim

        r = regularizers.l1_l2(10e-5)
        mlflow.log_param("regularizer", r)

        encoder_inputs = keras.Input(shape=(inputs_dim,))
        h1 = layers.Dense(inputs_dim, activation=self.activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=self.activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(inputs_dim / 3, activation=self.activation, activity_regularizer=r)(h2)

        z_mean = layers.Dense(self.latent_space_dimensions, name="z_mean")(h3)
        z_log_var = layers.Dense(self.latent_space_dimensions, name="z_log_var")(h3)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(self.latent_space_dimensions,))
        h1 = layers.Dense(inputs_dim / 3, activation=self.activation)(decoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=self.activation)(h1)

        decoder_outputs = layers.Dense(inputs_dim)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()
        mlflow.log_param("decoder_summary", self.decoder.summary())

        # Visualize the model.
        # tf.keras.utils.plot_model(model, to_file="model.png")

        # Train the VAE
        # Create the VAR, compile, and run.

        callback = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        self.history = self.vae.fit(self.data.X_train,
                                    validation_data=(self.data.X_val, self.data.X_val),
                                    epochs=30,
                                    callbacks=callback,
                                    batch_size=32,
                                    shuffle=True,
                                    verbose=1)

        save_path = Path("VAE", self.results_folder, "model")
        mlflow.keras.save_model(self.vae, save_path)
        mlflow.log_artifact(save_path)

    def predict(self):
        # Make some predictions
        cell = self.data.X_test[0]
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

    def calculate_r2_score(self):
        recon_test = self.vae.predict(self.data.X_test)
        recon_test = pd.DataFrame(data=recon_test, columns=self.data.markers)
        input_data = pd.DataFrame(data=self.data.X_test, columns=self.data.markers)

        # self.plot_clusters(input_data, range(len(self.data.markers)))

        for marker in self.data.markers:
            input_marker = input_data[f"{marker}"]
            var_marker = recon_test[f"{marker}"]

            self.r2_scores = self.r2_scores.append(
                {
                    "Marker": marker,
                    "Score": r2_score(input_marker, var_marker)
                }, ignore_index=True
            )
        # self.plot_label_clusters(self.data.X_test, self.data.X_test)

    def encode_decode_test_data(self):
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """
        mean, log_var, z = self.encoder.predict(self.data.X_test)
        self.encoded_data = pd.DataFrame(z)
        self.reconstructed_data = pd.DataFrame(columns=self.data.markers, data=self.decoder.predict(self.encoded_data))

        encoded_data_save_path = Path("VAE", self.results_folder, "encoded_data.csv")
        self.encoded_data.to_csv(encoded_data_save_path, index=False)
        mlflow.log_artifact(encoded_data_save_path)

        reconstructed_data_save_path = Path("VAE", self.results_folder, "reconstructed_data.csv")
        self.encoded_data.to_csv(reconstructed_data_save_path, index=False)
        mlflow.log_artifact(reconstructed_data_save_path)
