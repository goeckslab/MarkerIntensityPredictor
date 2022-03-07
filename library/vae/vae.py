from keras import layers, regularizers
import pandas as pd
import keras
from library.vae.sampling import Sampling
from library.vae.vae_model import VAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from library.vae.custom_callbacks import CustomCallback, WeightsForBatch


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class MarkerPredictionVAE:
    @staticmethod
    def build_variational_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                       input_dimensions: int,
                                       embedding_dimension: int, activation='relu'):
        """
        Sets up the vae and trains it
        """

        mlflow.tensorflow.autolog()

        r = regularizers.l1_l2(10e-5)
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
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        history = vae.fit(training_data,
                          validation_data=(validation_data, validation_data),
                          epochs=100,
                          callbacks=[early_stopping, WeightsForBatch()],
                          batch_size=256,
                          shuffle=True,
                          verbose=1)

        return vae, encoder, decoder, history
