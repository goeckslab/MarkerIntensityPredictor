import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.metrics import mean_squared_error, Mean
from keras.layers import Input, Dense, Lambda, Multiply, Concatenate
from typing import Tuple
from keras import backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from pathlib import Path
from library.data.folder_management import FolderManagement
from tensorflow.keras.utils import plot_model
from library.new_me_vae.sampling import SamplingLayer

# Important
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class NewMeVAE:
    def __init__(self, embedding_dimensions: int, marker_input_dimensions: int, morph_input_dimensions: int,
                 learning_rate: float, results_path: Path, **kwargs):
        self._results_path: Path = results_path
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
        decoder_output = self._decoder(Multiply()([self._marker_encoder.output[2], self._morph_encoder.output[2]]))

        self._vae = Model(inputs=[self._marker_encoder.input, self._morph_encoder.input],
                          outputs=[decoder_output], name="vae")
        self._vae.summary()
        plot_model(self._vae, Path.joinpath(self._results_path, "vae.png"))

        def marker_decoder_loss(true, pred):
            z_mean, z_log_var = self._marker_latent_space
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = reconstruction_loss_fn(true, pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + 0.001 * kl_loss
            return total_loss

        def morph_decoder_loss(true, pred):
            z_mean, z_log_var = self._morph_latent_space
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = reconstruction_loss_fn(true, pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + 0.001 * kl_loss
            return total_loss

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
              morph_val_data: pd.DataFrame, train_target_data: pd.DataFrame, val_target_data: pd.DataFrame):

        early_stopping = EarlyStopping(monitor="loss",
                                       mode="min", patience=5,
                                       restore_best_weights=True)

        csv_save_path: Path = Path.joinpath(self._results_path, "training.log")
        # if not csv_save_path.exists():
        csv_logger = CSVLogger(csv_save_path, separator=',')

        checkpoint_folder: Path = Path.joinpath(self._results_path, "checkpoints")

        if not checkpoint_folder.exists():
            FolderManagement.create_directory(checkpoint_folder)

        checkpoint = ModelCheckpoint(filepath=Path.joinpath(checkpoint_folder, "weights.hdf5"), verbose=1,
                                     save_weights_only=True, save_best_only=True)

        callbacks = [early_stopping, csv_logger, checkpoint]
        self._history = self._vae.fit(
            x={"marker_encoder_input": marker_train_data, "morph_encoder_input": morph_train_data},
            validation_data=[({"marker_encoder_input": marker_val_data, "morph_encoder_input": morph_val_data,
                               "decoder": val_target_data}),
                             ({"marker_encoder_input": marker_val_data, "morph_encoder_input": morph_val_data,
                               "decoder": val_target_data})],
            callbacks=callbacks, batch_size=192, epochs=250)

    def __build_marker_encoder(self, input_dimensions: int, layer_units: list):
        encoder_input = Input(shape=(input_dimensions,), name="marker_encoder_input")
        x = encoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_marker_encoder")(x)

        z_mean = Dense(self._embedding_dimensions, name='marker_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='marker_z_log_var')(x)

        z = SamplingLayer()([z_mean, z_log_var])

        self._marker_latent_space = (z_mean, z_log_var)
        self._marker_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="marker_encoder")
        self._marker_encoder.summary()
        plot_model(self._marker_encoder, Path.joinpath(self._results_path, "marker_encoder.png"))

    def __build_morph_encoder(self, input_dimensions: int, layer_units: list):
        encoder_input = Input(shape=(input_dimensions,), name="morph_encoder_input")
        x = encoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_morph_encoder")(x)

        z_mean = Dense(self._embedding_dimensions, name='morph_z_mean')(x)
        z_log_var = Dense(self._embedding_dimensions, name='morph_z_log_var')(x)

        z = SamplingLayer()([z_mean, z_log_var])

        self._morph_latent_space = (z_mean, z_log_var)
        self._morph_encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="morph_encoder")
        self._morph_encoder.summary()
        plot_model(self._morph_encoder, Path.joinpath(self._results_path, "morph_encoder.png"))

    def __build_decoder(self, layer_units: list):
        decoder_input = Input(shape=(self._embedding_dimensions,), name="decoder_input")
        x = decoder_input

        for i, layer_unit in enumerate(layer_units):
            x = Dense(units=layer_unit, activation='relu', name=f"{i}_decoder")(x)

        self._decoder = Model(inputs=decoder_input, outputs=x, name="decoder")
        self._decoder.summary()
        plot_model(self._decoder, Path.joinpath(self._results_path, "decoder.png"))

    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """

        z_mean, z_log_var = sample_args

        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self._embedding_dimensions),
                                  mean=0,
                                  stddev=1)

        return z_mean + K.exp(0.5 * z_log_var) * epsilon
