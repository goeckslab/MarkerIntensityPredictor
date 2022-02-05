from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import randrange
import pandas as pd
from pathlib import Path
import keras
from VAE.model.sampling import Sampling
from VAE.model.vae_model import VAE
from sklearn.metrics import r2_score
import tensorflow as tf
from Shared.data import Data
import mlflow
import mlflow.tensorflow


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class VAEModel:
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

    rmse = pd.DataFrame()

    # Latent space encoded data
    encoded_data = pd.DataFrame()

    # Reconstructed data from the latent space
    reconstructed_data = pd.DataFrame()

    args = None
    # The default results folder, where all experiment data is being stored.
    __base_result_path: Path
    # The sub folder used by mlflow
    __base_sub_folder = "VAE"

    def __init__(self, args, data: Data, base_results_path: Path, latent_space_dimensions=5,
                 activation="relu"):
        self.latent_space_dimensions = latent_space_dimensions
        self.args = args
        self.data = data
        self.activation = activation
        self.__base_result_path = base_results_path

        mlflow.log_param("input_dimensions", self.data.inputs_dim)
        mlflow.log_param("activation", self.activation)
        mlflow.log_param("latent_space_dimension", self.latent_space_dimensions)

    def build_auto_encoder(self):
        """

        """

        mlflow.tensorflow.autolog()

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
                                    epochs=100,
                                    callbacks=callback,
                                    batch_size=256,
                                    shuffle=True,
                                    verbose=1)

        save_path = Path(self.__base_result_path, "model")
        mlflow.keras.save_model(self.vae, save_path)
        mlflow.log_artifact(str(save_path), self.__base_sub_folder)

    def encode_decode_test_data(self):
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """
        mean, log_var, z = self.encoder.predict(self.data.X_test)
        self.encoded_data = pd.DataFrame(z)
        self.reconstructed_data = pd.DataFrame(columns=self.data.markers, data=self.decoder.predict(self.encoded_data))

        encoded_data_save_path = Path(self.__base_result_path, "encoded_data.csv")
        self.encoded_data.to_csv(encoded_data_save_path, index=False)
        mlflow.log_artifact(str(encoded_data_save_path), self.__base_sub_folder)

        reconstructed_data_save_path = Path(self.__base_result_path, "reconstructed_data.csv")
        self.encoded_data.to_csv(reconstructed_data_save_path, index=False)
        mlflow.log_artifact(str(reconstructed_data_save_path), self.__base_sub_folder)
