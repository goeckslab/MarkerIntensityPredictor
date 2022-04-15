from keras import layers, regularizers
import pandas as pd
import keras
from library.me_vae.sampling import Sampling
from keras.layers import concatenate
from library.me_vae.me_vae_model import MEVAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from library.me_vae.custom_callbacks import CustomCallback, WeightsForBatch
from tensorflow.keras.models import Model
from typing import Tuple


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class MEMarkerPredictionVAE:

    @staticmethod
    def build_me_variational_auto_encoder(training_data: Tuple,
                                          validation_data: Tuple,
                                          output_dimensions: int,
                                          embedding_dimension: int, activation='relu',
                                          learning_rate: float = 1e-3,
                                          amount_of_layers: int = 5,
                                          optimizer: str = "adam",
                                          use_ml_flow: bool = True):
        """
        Sets up a 5 layer vae and trains it
        """

        if len(training_data) != 2:
            raise ValueError("Training and validation data must contain two datasets!")

        marker_training_data = training_data[0]
        morph_training_data = training_data[1]

        marker_val_data = validation_data[0]
        morph_val_data = validation_data[1]

        if use_ml_flow:
            mlflow.tensorflow.autolog()

        r = regularizers.l1_l2(10e-5)

        # Switch network when layers are redefined
        if amount_of_layers == 5:
            marker_nn = MEMarkerPredictionVAE.__create_marker_nn_5_layers(marker_training_data.shape[1], activation, r)
        else:
            marker_nn = MEMarkerPredictionVAE.__create_marker_nn_3_layers(marker_training_data.shape[1], activation, r)

        morph_nn = MEMarkerPredictionVAE.__create_morpho_nn(morph_training_data.shape[1], activation, r)
        combined_input = concatenate([marker_nn.output, morph_nn.output])

        # Latent space
        z_mean = layers.Dense(embedding_dimension, name="z_mean")(combined_input)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(combined_input)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=[marker_nn.input, morph_nn.input], outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(embedding_dimension,))
        h1 = layers.Dense(output_dimensions / 3, activation=activation, name="decoding_h1")(decoder_inputs)
        h2 = layers.Dense(output_dimensions / 2.5, activation=activation, name="decoding_h2")(h1)
        h3 = layers.Dense(output_dimensions / 2, activation=activation, name="decoding_h3")(h2)
        h4 = layers.Dense(output_dimensions / 1.5, activation=activation, name="decoding_h4")(h3)

        decoder_outputs = layers.Dense(output_dimensions, name="decoder_output")(h4)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        me_vae = MEVAE(encoder, decoder)
        me_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        history = me_vae.fit([marker_training_data, morph_training_data],
                             validation_data=([marker_val_data, morph_val_data], [marker_val_data, morph_val_data]),
                             epochs=500,
                             callbacks=[early_stopping, WeightsForBatch()],
                             batch_size=256,
                             shuffle=True,
                             verbose=1)

        return me_vae, encoder, decoder, history

    @staticmethod
    def __create_marker_nn_5_layers(input_dimensions: int, activation: str, r: int):
        """
        Create the model for the markers
        @param input_dimensions:
        @param activation:
        @param r:
        @return:
        """
        encoder_inputs = keras.Input(shape=(input_dimensions,))
        h1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="marker_1")(
            encoder_inputs)
        h2 = layers.Dense(input_dimensions / 1.5, activation=activation, activity_regularizer=r, name="marker_2")(h1)
        h3 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="marker_3")(h2)
        h4 = layers.Dense(input_dimensions / 2.5, activation=activation, activity_regularizer=r, name="marker_4")(h3)
        h5 = layers.Dense(input_dimensions / 3, activation=activation, activity_regularizer=r, name="marker_5")(h4)

        model = Model(encoder_inputs, h5)

        return model

    @staticmethod
    def __create_marker_nn_3_layers(input_dimensions: int, activation: str, r: int):
        """
        Create the model for the markers
        @param input_dimensions:
        @param activation:
        @param r:
        @return:
        """
        encoder_inputs = keras.Input(shape=(input_dimensions,))
        h1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="marker_1")(
            encoder_inputs)
        h2 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="marker_2")(h1)
        h3 = layers.Dense(input_dimensions / 3, activation=activation, activity_regularizer=r, name="marker_3")(h2)

        model = Model(encoder_inputs, h3)

        return model

    @staticmethod
    def __create_morpho_nn(input_dimensions, activation: str, r: int):
        encoder_inputs = keras.Input(shape=(input_dimensions,))
        g1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="morph_1")(
            encoder_inputs)
        g2 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="morph_2")(g1)

        model = Model(encoder_inputs, g2)

        return model
