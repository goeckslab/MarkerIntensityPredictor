import keras
from keras import layers, regularizers
import tensorflow as tf
from entities.data import Data
from vae_model.sampling import Sampling
from vae_model.vae import VAE
import numpy as np
import pandas as pd


class VariationalAutoEncoder:
    activation = "relu"
    activity_regularizer = regularizers.l1_l2(10e-5)

    # The dimensions of the latent space
    latent_space_dimension: int = 2

    # Encoder part of the network
    encoder = None

    # Decoder part of the network
    decoder = None

    # Neurons before flatten
    nbl: np.ndarray

    # The combined model
    vae = None
    # The train history
    history = None

    input_layer = None
    output_layer = None

    data: Data = None

    def __init__(self, data: Data):
        self.data = data

    def build_auto_encoder(self):
        # Build the encoder
        inputs_dim = self.data.inputs_dim
        r = regularizers.l1_l2(10e-5)
        activation = tf.keras.layers.ReLU()

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

        z_mean = layers.Dense(self.latent_space_dimension, name="z_mean")(lbf)
        z_log_var = layers.Dense(self.latent_space_dimension, name="z_log_var")(lbf)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(self.latent_space_dimension,))
        h1 = layers.Dense(nbl, activation=activation)(decoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation)(h1)

        decoder_outputs = layers.Dense(inputs_dim)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()
        # Visualize the model.
        # tf.keras.utils.plot_model(model, to_file="model.png")

        # Train the VAE
        # Create the VAR, compile, and run.

        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

        callback = keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                 mode="min", patience=5,
                                                 restore_best_weights=True)
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam())

        # self.history = self.vae.fit(mnist_digits, epochs=5, batch_size=128,
        #                            verbose=1, callbacks=callback)  #
        self.history = self.vae.fit(self.data.X_train,
                                    epochs=100,
                                    callbacks=callback,
                                    validation_data=(self.data.X_val, self.data.X_val),
                                    batch_size=128,
                                    shuffle=True,
                                    verbose=1)

    def predict(self):
        # Make some predictions
        cell = self.data.X_test[0]
        cell = cell.reshape(1, cell.shape[0])
        mean, log_var, z = self.encoder.predict(cell)
        encoded_cell = z
        decoded_cell = self.decoder.predict(encoded_cell)
        return cell, encoded_cell, decoded_cell

    def reconstruction(self):
        return self.vae.predict(self.data.X_test), self.data.X_test
