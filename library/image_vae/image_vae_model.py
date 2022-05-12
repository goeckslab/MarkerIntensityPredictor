import pandas as pd
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.layers import concatenate, Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, Lambda, multiply
from keras import metrics
from typing import Tuple
from keras import backend as K


class ImageVAE(Model):

    def __init__(self, embedding_dimensions: int, n_channels: int, filters: int, kernel_size: Tuple,
                 learning_rate: float):
        super(ImageVAE, self).__init__()
        self._embedding_dimensions: int = embedding_dimensions
        self._feature_encoder: Model = None
        self._feature_decoder: Model = None
        self._image_encoder: Model = None
        self._image_decoder: Model = None
        self._vae: Model = None

        self._history = None

        # Amount of channels in image
        self._n_channels: int = n_channels
        self._filters: int = filters
        self._kernel_size: Tuple = kernel_size
        self._learning_rate = learning_rate
        self._encoded_image_shape: Tuple = ()

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

        self._feature_latent_space: Tuple = ()
        self._image_latent_space: Tuple = ()

        self.build_feature_encoder(input_dimensions=25, layer_units=[12, 9, 7])
        self.build_feature_decoder(layer_units=[7, 9, 12, 25])
        self.build_image_encoder(input_dimensions=(640, 480), layers=2)
        self.build_image_decoder(layers=2)

    @property
    def image_encoder(self):
        return self._image_encoder

    @property
    def feature_encoder(self):
        return self._feature_encoder

    @property
    def feature_decoder(self):
        return self._feature_decoder

    @property
    def image_decoder(self):
        return self._image_decoder

    @property
    def vae(self):
        return self._vae

    def build_model(self):

        shared_latent_space = multiply([self._feature_encoder.output[2], self._image_encoder.output[2]])
        decoder_output = self._feature_decoder(shared_latent_space)
        image_output = self._image_decoder(shared_latent_space)

        self._vae = Model(inputs=[self._feature_encoder.input, self._image_encoder.input],
                          outputs=[decoder_output, image_output],
                          name="vae")
        self._vae.summary()

        # me_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        losses = {"image_decoder": self.image_decoder_loss, "feature_decoder": self.feature_decoder_loss}
        # lossWeights = {"decoder1": 1.0}

        self._vae.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate))

    def train(self, feature_train_data: pd.DataFrame, feature_val_data: pd.DataFrame, image_train_data: pd.DataFrame,
              image_val_data: pd.DataFrame):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)

        callbacks = [early_stopping]
        self._history = self._vae.fit(
            x={"feature_encoder_input": feature_train_data, "image_encoder_input": image_train_data},
            validation_data=([feature_val_data, image_val_data], [feature_val_data, image_val_data]),
            callbacks=callbacks, batch_size=192, epochs=100)

    def build_feature_encoder(self, input_dimensions: int, layer_units: list):
        encoder_input = Input(shape=(input_dimensions,), name="feature_encoder_input")
        x = encoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_feature_encoder")(x)

        z_mean = Dense(self._embedding_dimensions, name='feature_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='feature_z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self._embedding_dimensions,), name='z_feature_encoder')(
            [z_mean, z_log_var])

        self._feature_latent_space = (z_mean, z_log_var)
        self._feature_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="feature_encoder")
        self._feature_encoder.summary()

    def build_image_encoder(self, input_dimensions: Tuple, layers: int):
        encoder_input = Input(shape=(input_dimensions[0], input_dimensions[1], self._n_channels),
                              name="image_encoder_input")

        x = encoder_input
        for i in range(layers):
            x = Conv2D(filters=self._filters,
                       kernel_size=self._kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same',
                       name=f"{i}_image_encoder")(x)
            self._filters *= 2

        self._encoded_image_shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(self._embedding_dimensions, activation='relu')(x)

        z_mean = Dense(self._embedding_dimensions, name='image_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='image_z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self._embedding_dimensions,), name='z_image_encoder')(
            [z_mean, z_log_var])

        self._image_latent_space = (z_mean, z_log_var)
        self._image_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="image_encoder")
        self._image_encoder.summary()

    def build_feature_decoder(self, layer_units: list):
        feature_decoder_input = Input(shape=(self._embedding_dimensions,), name="feature_decoder_input")
        x = feature_decoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_feature_decoder")(x)

        self._feature_decoder = Model(inputs=feature_decoder_input, outputs=x, name="feature_decoder")
        self._feature_decoder.summary()

    def build_image_decoder(self, layers: int):
        # build decoder model
        image_decoder_input = Input(shape=(self._embedding_dimensions,), name='image_decoder_input')

        shape = self._encoded_image_shape[1] * self._encoded_image_shape[2] * self._encoded_image_shape[3]
        x = Dense(units=shape, activation='relu', name="dense_layer")(image_decoder_input)
        x = Reshape((self._encoded_image_shape[1], self._encoded_image_shape[2], self._encoded_image_shape[3]))(x)

        for i in range(layers):
            x = Conv2DTranspose(filters=self._filters,
                                kernel_size=self._kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same',
                                name=f"{i}_image_decoder")(x)
            self._filters //= 2

        x = Conv2DTranspose(filters=self._filters,
                            kernel_size=self._kernel_size,
                            activation='relu',
                            padding='same',
                            name='decoder_output')(x)

        self._image_decoder = Model(inputs=image_decoder_input, outputs=x, name='image_decoder')
        self._image_decoder.summary()

    def feature_decoder_loss(self, true, pred):
        z_mean, z_log_var = self._feature_latent_space
        recon_loss = metrics.mean_squared_error(true, pred)
        recon_loss *= self.image_size * self.image_size
        kl_loss = 1 + z_log_var * 2 - K.square(z_mean) - K.exp(z_log_var * 2)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(recon_loss + kl_loss)
        return vae_loss / 2

    def image_decoder_loss(self, true, pred):
        z_mean, z_log_var = self._image_latent_space
        recon_loss = metrics.binary_crossentropy(K.flatten(true), K.flatten(pred))
        recon_loss *= self.image_size * self.image_size
        kl_loss = 1 + z_log_var * 2 - K.square(z_mean) - K.exp(z_log_var * 2)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(recon_loss + kl_loss)
        return vae_loss / 2

    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """

        z_mean, z_log_var = sample_args

        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self._embedding_dimensions),
                                  mean=0,
                                  stddev=1)

        return z_mean + K.exp(0.5 * z_log_var) * epsilon
