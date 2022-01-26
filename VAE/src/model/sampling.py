from keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# Create sampling layer.
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a cell."""

    @staticmethod
    def call(inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        epsilon = tfp.distributions.MultivariateNormalDiag(
            loc=[1., -1],
            scale_diag=[1, 2.],
            name='MultivariateNormalDiag'
        )

        epsilon = epsilon.prob([-1., 1])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
