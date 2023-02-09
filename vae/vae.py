import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Layer
import argparse


class BiModalSampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.uniform(shape=tf.shape(z_mean), minval=0, maxval=1)
        std = tf.exp(0.5 * z_log_var)
        z = z_mean + std * tf.where(tf.less(epsilon, 0.5), tf.ones_like(z_mean), -tf.ones_like(z_mean))
        return z


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss_fn = keras.losses.MeanAbsoluteError()
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("-trb", "--train_biopsy", help="Train Biopsy", required=True)
    parser.add_argument("-tb", "--test_biopsy", help="Test Biopsy", required=True)
    args = parser.parse_args()

    train_biopsy = pd.read_csv(args.train_biopsy, delimiter="\t")
    test_biopsy = pd.read_csv(args.test_biopsy, delimiter="\t")

    latent_dim = 5
    print(train_biopsy.shape)
    encoder_inputs = keras.Input(shape=(train_biopsy.shape[1],))
    x = Dense(16, activation="relu")(encoder_inputs)
    x = Dense(8, activation="relu")(x)
    x = Dense(latent_dim, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])

    z = BiModalSampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim, activation="relu")(latent_inputs)
    x = Dense(8, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    decoder_outputs = Dense(train_biopsy.shape[1])(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit(train_biopsy, epochs=30, batch_size=128)

    predicted = pd.DataFrame(vae.decoder.predict(vae.encoder.predict(test_biopsy)[2]), columns=test_biopsy.columns)
    print(predicted)
    # calucate the mean absolute error for each column
    for col in predicted.columns:
        print(col, np.mean(np.abs(predicted[col] - test_biopsy[col])))
