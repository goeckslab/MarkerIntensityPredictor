from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np


class NewModel(Model):

    def __init__(self, marker_encoder, morph_encoder, decoder, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        self._marker_encoder = marker_encoder
        self._morph_encoder = morph_encoder
        self._decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            marker_z_mean, marker_z_log_var, marker_z = self._marker_encoder(data[0][0])
            reconstruction = self.decoder(marker_z)
            reconstruction_loss_fn = MeanSquaredError()
            reconstruction_loss = reconstruction_loss_fn(data[0][0], reconstruction)
            kl_loss = -0.5 * (1 + marker_z_log_var - tf.square(marker_z_mean) - tf.exp(marker_z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            marker_total_loss = reconstruction_loss + 0.001 * kl_loss

            morph_z_mean, morph_z_log_var, morph_z = self._morph_encoder(data[0][1])
            reconstruction = self.decoder(morph_z)
            reconstruction_loss_fn = MeanSquaredError()
            reconstruction_loss = reconstruction_loss_fn(data[0][1], reconstruction)
            kl_loss = -0.5 * (1 + morph_z_log_var - tf.square(morph_z_mean) - tf.exp(morph_z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            morph_total_loss = reconstruction_loss + 0.001 * kl_loss

            combined_loss = marker_total_loss + morph_total_loss

        grads = tape.gradient(combined_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(combined_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        # inputs = tf.reshape(inputs, shape=(1, 25))

        marker_indices = range(0, 20)
        morph_indices = range(20, 25)

        if type(inputs) is tuple:
            marker = inputs[0]
            morph = inputs[1]
        else:
            marker = tf.gather(inputs, marker_indices, axis=1)
            morph = tf.gather(inputs, morph_indices, axis=1)
            #marker = inputs.get('marker_encoder_input')
            #morph = inputs.get('morph_encoder_input')

        _, _, marker_z = self._marker_encoder(marker.to_numpy())
        _, _, morph_z = self._morph_encoder(morph.to_numpy())
        z = np.multiply(marker_z, morph_z)
        return self._decoder(z)
