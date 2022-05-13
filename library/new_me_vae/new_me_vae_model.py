import pandas as pd
import tensorflow
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.losses import MeanSquaredError, Loss
from keras.metrics import mean_squared_error, Mean
from keras.layers import concatenate, Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, Lambda, Multiply
from typing import Tuple, Dict
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class NewMeVAE:
    def __init__(self, embedding_dimensions: int, marker_input_dimensions: int, morph_input_dimensions: int,
                 learning_rate: float):
        self._embedding_dimensions: int = embedding_dimensions
        self._marker_encoder: Model = None
        self._morph_encoder: Model = None
        self._decoder: Model = None
        self._vae: Model = None

        self._history = None

        self._learning_rate = learning_rate

        self._marker_latent_space: Tuple = ()
        self._morph_latent_space: Tuple = ()

        self.__build_marker_encoder(input_dimensions=marker_input_dimensions, layer_units=[25, 12, 9, 7])
        self.__build_morph_encoder(input_dimensions=morph_input_dimensions, layer_units=[5, 5])
        self.__build_decoder(layer_units=[7, 9, 12, 25])

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def morph_encoder(self):
        return self._morph_encoder

    @property
    def marker_encoder(self):
        return self._marker_encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def vae(self):
        return self._vae

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def build_model(self):

        decoder_output = self.decoder(Multiply()([self._marker_encoder.output[2], self._morph_encoder.output[2]]))

        self._vae = Model(inputs=[self._marker_encoder.input, self._morph_encoder.input],
                          outputs=[decoder_output],
                          name="vae")
        self._vae.summary()

        def marker_decoder_loss(true, pred):
            z_mean, z_log_var = self._marker_latent_space
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = mean_squared_error(true, pred)
            kl_loss = 1 + z_log_var * 2 - K.square(z_mean) - K.exp(z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            vae_loss = K.mean(recon_loss + kl_loss)
            return vae_loss / 2

        def morph_decoder_loss(true, pred):
            z_mean, z_log_var = self._morph_latent_space
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = mean_squared_error(true, pred)
            kl_loss = 1 + z_log_var * 2 - K.square(z_mean) - K.exp(z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            vae_loss = K.mean(recon_loss + kl_loss)
            return vae_loss / 2

        def combined_loss(true, pred):
            marker_loss = marker_decoder_loss(true, pred)
            morph_loss = morph_decoder_loss(true, pred)
            return marker_loss + morph_loss

        # me_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        losses = {"decoder": combined_loss}
        loss_weights = {"decoder": 1.0}

        self._vae.compile(loss=losses, loss_weights=loss_weights,
                          optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate))

    def train(self, marker_train_data: pd.DataFrame, marker_val_data: pd.DataFrame, morph_train_data: pd.DataFrame,
              morph_val_data: pd.DataFrame, target_data: pd.DataFrame):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)

        callbacks = [early_stopping]
        self._history = self._vae.fit(
            x={"marker_encoder_input": marker_train_data, "morph_encoder_input": morph_train_data}, y=target_data,
            callbacks=callbacks, batch_size=192, epochs=100)

    def __build_marker_encoder(self, input_dimensions: int, layer_units: list):
        encoder_input = Input(shape=(input_dimensions,), name="marker_encoder_input")
        x = encoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_marker_encoder")(x)

        z_mean = Dense(self._embedding_dimensions, name='marker_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='marker_z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self._embedding_dimensions,), name='z_marker_encoder')(
            [z_mean, z_log_var])

        self._marker_latent_space = (z_mean, z_log_var)
        self._marker_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="marker_encoder")
        self._marker_encoder.summary()

    def __build_morph_encoder(self, input_dimensions: int, layer_units: list):
        encoder_input = Input(shape=(input_dimensions,), name="morph_encoder_input")
        x = encoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_morph_encoder")(x)

        z_mean = Dense(self._embedding_dimensions, name='morph_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='morph_z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self._embedding_dimensions,), name='z_morph_encoder')(
            [z_mean, z_log_var])

        self._morph_latent_space = (z_mean, z_log_var)
        self._morph_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="morph_encoder")
        self._morph_encoder.summary()

    def __build_decoder(self, layer_units: list):
        decoder_input = Input(shape=(self._embedding_dimensions,), name="decoder_input")
        x = decoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_decoder")(x)

        self._decoder = Model(inputs=decoder_input, outputs=x, name="decoder")
        self._decoder.summary()

    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """

        z_mean, z_log_var = sample_args

        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self._embedding_dimensions),
                                  mean=0,
                                  stddev=1)

        return z_mean + K.exp(0.5 * z_log_var) * epsilon
