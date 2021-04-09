from keras import layers, regularizers
import keras
import numpy as np
from VAE.entities.sampling import Sampling


class Encoder:
    @staticmethod
    def build_encoder(inputs):
        """
        Builds the encoder
        :param inputs:
        :return:
        """
        # Build the encoder
        # length of latent vector.
        # TODO: Add cli arg to play around
        latent_dim = 6

        inputs_dim = inputs.shape[1]
        r = regularizers.l1(10e-5)
        activation = "linear"

        encoder_inputs = keras.Input(shape=(inputs_dim))
        h1 = layers.Dense(inputs_dim, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(inputs_dim / 3, activation=activation, activity_regularizer=r)(h2)

        # The following variables are for the convenience of building the decoder.
        # last layer before flatten
        lbf = h3
        # shape before flatten.
        sbf = keras.backend.int_shape(lbf)[1:]
        # neurons count before latent dim
        nbl = np.prod(sbf)

        z_mean = layers.Dense(latent_dim, name="z_mean")(lbf)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(lbf)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
