from pathlib import Path
from Shared.data import Data
import keras
from keras import layers, regularizers
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
import mlflow
from keras.losses import MeanAbsoluteError, MeanSquaredError


class AutoEncoderModel:
    # The data used to train and evaluate the model
    __data: Data

    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    ae: any

    # the training history of the AE
    history: any

    latent_space_dimensions: int

    # The encoded data
    encoded_data = pd.DataFrame()
    # The reconstructed data
    reconstructed_data = pd.DataFrame()
    args = None

    # the base results folder
    __base_result_path: Path

    # The sub folder used by mlflow
    __base_sub_folder = "AE"

    # The compiled model
    model = None

    # The activation used for the auto encoder
    activation: str

    def __init__(self, args, data: Data, base_result_path: Path, latent_space_dimensions=5,
                 activation='relu'):
        self.__base_result_path = base_result_path
        self.latent_space_dimensions = latent_space_dimensions
        self.__data = data
        self.args = args
        self.activation = activation

        mlflow.log_param("input_dimensions", self.__data.inputs_dim)
        mlflow.log_param("activation", self.activation)
        mlflow.log_param("latent_space_dimension", self.latent_space_dimensions)

    def build_auto_encoder(self):
        # activate auto log
        mlflow.tensorflow.autolog()

        # Build encoder
        input_layers = keras.Input(shape=(self.__data.inputs_dim,))
        encoded = layers.Dense(self.__data.inputs_dim, activation=self.activation, name="encoding_h1")(input_layers)
        encoded = layers.Dense(self.__data.inputs_dim / 2, activation=self.activation, name="encoding_h2")(encoded)
        encoded = layers.Dense(self.__data.inputs_dim / 3, activation=self.activation, name="encoding_h3")(encoded)
        encoded = layers.Dense(self.latent_space_dimensions, activation=self.activation, name="embedding")(encoded)

        # Build decoder.
        decoded = layers.Dense(self.__data.inputs_dim / 3, activation=self.activation, name="decoding_h1")(encoded)
        decoded = layers.Dense(self.__data.inputs_dim / 2, activation=self.activation, name="decoding_h2")(decoded)
        decoded = layers.Dense(self.__data.inputs_dim, name="decoder_output")(decoded)

        # Auto encoder
        self.ae = keras.Model(input_layers, decoded, name="AE")
        self.ae.summary()

        # Separate encoder model
        self.encoder = keras.Model(input_layers, encoded, name="encoder")
        self.encoder.summary()

        # Separate decoder model
        encoded_input = keras.Input(shape=(self.latent_space_dimensions,))
        deco = self.ae.layers[-3](encoded_input)
        deco = self.ae.layers[-2](deco)
        deco = self.ae.layers[-1](deco)
        # create the decoder model
        self.decoder = keras.Model(encoded_input, deco, name="decoder")
        self.decoder.summary()

        # Compile ae
        self.ae.compile(optimizer="adam", loss=MeanAbsoluteError())

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.history = self.ae.fit(self.__data.X_train, self.__data.X_train,
                                   epochs=100,
                                   batch_size=256,
                                   shuffle=True,
                                   callbacks=[callback],
                                   validation_data=(self.__data.X_val, self.__data.X_val))

    def encode_decode_test_data(self):
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """
        encoded = self.encoder.predict(self.__data.X_test)
        self.encoded_data = pd.DataFrame(encoded)
        self.reconstructed_data = pd.DataFrame(columns=self.__data.markers, data=self.decoder.predict(self.encoded_data))

        encoded_data_save_path = Path(self.__base_result_path, "encoded_data.csv")
        self.encoded_data.to_csv(encoded_data_save_path, index=False)
        mlflow.log_artifact(str(encoded_data_save_path), self.__base_sub_folder)

        reconstructed_data_save_path = Path(self.__base_result_path, "reconstructed_data.csv")
        self.encoded_data.to_csv(reconstructed_data_save_path, index=False)
        mlflow.log_artifact(str(reconstructed_data_save_path), self.__base_sub_folder)
