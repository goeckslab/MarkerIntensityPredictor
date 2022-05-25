import tensorflow as tf
from tensorflow import keras
from keras.layers import concatenate
from tensorflow.keras.models import Model


class NewModel(keras.Model):
    def __init__(self, marker_encoder: Model, morph_encoder: Model, decoder: Model, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        self.marker_encoder = marker_encoder
        self.morph_encoder = morph_encoder
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

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:

            marker_loss, marker_kl = self.__calculate_loss(self.marker_encoder, self.decoder, )


            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss_fn = keras.losses.MeanSquaredError()
            data = concatenate([data[0][0], data[0][1]])
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + 0.001 * kl_loss

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

        z_mean, z_log_var, z = self.encoder([marker, morph])
        return self.decoder(z)

    def __calculate_loss(self, encoder: Model, decoder: Model, data) -> Tuple:
        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        reconstruction_loss_fn = keras.losses.MeanSquaredError()

        # Calculate los
        recon_loss = reconstruction_loss_fn(data, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return recon_loss, kl_loss
