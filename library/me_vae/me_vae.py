from keras import layers, regularizers
import pandas as pd
import keras
from library.me_vae.sampling import Sampling
from keras.layers import concatenate
from library.me_vae.vae_model import VAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from library.me_vae.custom_callbacks import CustomCallback, WeightsForBatch
from tensorflow.keras.models import Model


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class MEMarkerPredictionVAE:
    def build_me_variational_auto_encoder(self, marker_training_data: pd.DataFrame,
                                          morph_training_data: pd.DataFrame,
                                          validation_data: pd.DataFrame,
                                          input_dimensions: int,
                                          embedding_dimension: int, activation='relu',
                                          learning_rate: float = 1e-3,
                                          amount_of_layers: int = 5,
                                          optimizer: str = "adam",
                                          use_ml_flow: bool = True):
        """
            Sets up a 5 layer vae and trains it
            """

        if use_ml_flow:
            mlflow.tensorflow.autolog()

        r = regularizers.l1_l2(10e-5)

        if use_ml_flow:
            mlflow.log_param("regularizer", r)

        # Switch network when layers are redefined
        if amount_of_layers == 5:
            marker_nn = self.create_marker_nn_5_layers(marker_training_data.shape[1], activation, r)
        else:
            marker_nn = self.create_marker_nn_3_layers(marker_training_data.shape[1], activation, r)

        morph_nn = self.create_morpho_nn(morph_training_data.shape[1], activation, r)
        combined_input = concatenate([marker_nn.output, morph_nn.output])

        # Latent space
        z_mean = layers.Dense(embedding_dimension, name="z_mean")(combined_input)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(combined_input)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=[marker_nn.input, morph_nn.input], outputs=[z_mean, z_log_var, z], name="encoder")
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
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        history = vae.fit([marker_training_data, morph_training_data],
                          validation_data=(validation_data, validation_data),
                          epochs=500,
                          callbacks=[early_stopping, WeightsForBatch()],
                          batch_size=256,
                          shuffle=True,
                          verbose=1)

        return vae, encoder, decoder, history

    def create_marker_nn_5_layers(self, input_dimensions: int, activation: str, r: int):
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

    def create_marker_nn_3_layers(self, input_dimensions: int, activation: str, r: int):
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

    def create_morpho_nn(self, input_dimensions, activation: str, r: int):
        encoder_inputs = keras.Input(shape=(input_dimensions,))
        g1 = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r, name="morph_1")(
            encoder_inputs)
        g2 = layers.Dense(input_dimensions / 2, activation=activation, activity_regularizer=r, name="morph_2")(g1)

        model = Model(encoder_inputs, g2)

        return model
