from keras import layers, regularizers
import pandas as pd
import keras
from library.vae.sampling import Sampling
from keras.layers import concatenate
from library.vae.vae_model import VAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from library.vae.custom_callbacks import CustomCallback, WeightsForBatch


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class MarkerPredictionVAE:
    @staticmethod
    def build_variational_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                       input_dimensions: int, embedding_dimension: int, activation='relu',
                                       learning_rate: float = 1e-3,
                                       optimizer: str = "adam", use_ml_flow: bool = True, amount_of_layers: int = 3):
        """
        Handles which vae to build
        @param training_data:
        @param validation_data:
        @param input_dimensions:
        @param embedding_dimension:
        @param activation:
        @param learning_rate:
        @param optimizer:
        @param use_ml_flow:
        @param amount_of_layers:
        @return:
        """
        if amount_of_layers == 3:
            return MarkerPredictionVAE.build_3_layer_variational_auto_encoder(training_data=training_data,
                                                                              validation_data=validation_data,
                                                                              input_dimensions=input_dimensions,
                                                                              embedding_dimension=embedding_dimension,
                                                                              activation=activation,
                                                                              learning_rate=learning_rate,
                                                                              optimizer=optimizer,
                                                                              use_ml_flow=use_ml_flow)

        elif amount_of_layers == 5:
            return MarkerPredictionVAE.build_5_layer_variational_auto_encoder(training_data=training_data,
                                                                              validation_data=validation_data,
                                                                              input_dimensions=input_dimensions,
                                                                              embedding_dimension=embedding_dimension,
                                                                              activation=activation,
                                                                              learning_rate=learning_rate,
                                                                              optimizer=optimizer,
                                                                              use_ml_flow=use_ml_flow)

        else:
            raise ValueError("Amount of layers only supports 3 or 5 layers currently")

    @staticmethod
    def build_3_layer_variational_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                               input_dimensions: int,
                                               embedding_dimension: int, activation='relu', learning_rate: float = 1e-3,
                                               optimizer: str = "adam", use_ml_flow: bool = True):
        """
        Sets up a 3 layer vae and trains it
        """

        if use_ml_flow:
            mlflow.tensorflow.autolog()

        r = regularizers.l1_l2(10e-5)

        if use_ml_flow:
            mlflow.log_param("regularizer", r)

        encoder_inputs = keras.Input(shape=(input_dimensions,))
        h1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="encoding_h1")(
            encoder_inputs)
        h2 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="encoding_h2")(h1)
        h3 = layers.Dense(input_dimensions / 3, activation=activation, activity_regularizer=r, name="encoding_h3")(h2)

        z_mean = layers.Dense(embedding_dimension, name="z_mean")(h3)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(h3)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(embedding_dimension,))
        h1 = layers.Dense(input_dimensions / 3, activation=activation, name="decoding_h1")(decoder_inputs)
        h2 = layers.Dense(input_dimensions / 2, activation=activation, name="decoding_h2")(h1)

        decoder_outputs = layers.Dense(input_dimensions, name="decoder_output")(h2)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        vae = VAE(encoder, decoder)

        if optimizer == "adam":
            vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        elif optimizer == "sgd":
            vae.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
        else:
            raise ValueError("Please specify a valid optimizer.")

        history = vae.fit(training_data,
                          validation_data=(validation_data, validation_data),
                          epochs=500,
                          callbacks=[early_stopping, WeightsForBatch()],
                          batch_size=256,
                          shuffle=True,
                          verbose=1)

        return vae, encoder, decoder, history

    @staticmethod
    def build_5_layer_variational_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                               input_dimensions: int,
                                               embedding_dimension: int, activation='relu', learning_rate: float = 1e-3,
                                               optimizer: str = "adam", use_ml_flow: bool = True):
        """
        Sets up a 5 layer vae and trains it
        """

        if use_ml_flow:
            mlflow.tensorflow.autolog()

        r = regularizers.l1_l2(10e-5)

        if use_ml_flow:
            mlflow.log_param("regularizer", r)

        encoder_inputs = keras.Input(shape=(input_dimensions,))
        h1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="encoding_h1")(
            encoder_inputs)
        h2 = layers.Dense(input_dimensions / 1.5, activation=activation, activity_regularizer=r, name="encoding_h2")(h1)
        h3 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="encoding_h3")(h2)
        h4 = layers.Dense(input_dimensions / 2.5, activation=activation, activity_regularizer=r, name="encoding_h4")(h3)
        h5 = layers.Dense(input_dimensions / 3, activation=activation, activity_regularizer=r, name="encoding_h5")(h4)

        # Latent space
        z_mean = layers.Dense(embedding_dimension, name="z_mean")(h5)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(h5)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(embedding_dimension,))
        h1 = layers.Dense(input_dimensions / 3, activation=activation, name="decoding_h1")(decoder_inputs)
        h2 = layers.Dense(input_dimensions / 2.5, activation=activation, name="decoding_h2")(h1)
        h3 = layers.Dense(input_dimensions / 2, activation=activation, name="decoding_h3")(h2)
        h4 = layers.Dense(input_dimensions / 1.5, activation=activation, name="decoding_h4")(h3)

        decoder_outputs = layers.Dense(input_dimensions, name="decoder_output")(h4)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        vae = VAE(encoder, decoder)

        if optimizer == "adam":
            vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        elif optimizer == "sgd":
            vae.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
        else:
            raise ValueError("Please specify a valid optimizer.")

        history = vae.fit(training_data,
                          validation_data=(validation_data, validation_data),
                          epochs=500,
                          callbacks=[early_stopping, WeightsForBatch()],
                          batch_size=256,
                          shuffle=True,
                          verbose=1)

        return vae, encoder, decoder, history