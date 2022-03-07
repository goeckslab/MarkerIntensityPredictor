import keras
import mlflow
import pandas as pd
from keras import layers
import tensorflow as tf
from keras.losses import MeanAbsoluteError


class AutoEncoder:

    @staticmethod
    def build_auto_encoder(train_data: pd.DataFrame, validation_data: pd.DataFrame, input_dimensions: int,
                           embedding_dimensions: int, activation='relu'):
        """

        @param train_data:
        @param validation_data:
        @param input_dimensions:
        @param embedding_dimensions:
        @param activation:
        @return: Returns the model, encoder, decoder and training history
        """
        # activate auto log
        mlflow.tensorflow.autolog()

        # Build encoder
        input_layers = keras.Input(shape=(input_dimensions,))
        encoded = layers.Dense(input_dimensions, activation=activation, name="encoding_h1")(input_layers)
        encoded = layers.Dense(input_dimensions / 2, activation=activation, name="encoding_h2")(encoded)
        encoded = layers.Dense(input_dimensions / 3, activation=activation, name="encoding_h3")(encoded)
        encoded = layers.Dense(embedding_dimensions, activation=activation, name="embedding")(encoded)

        # Build decoder.
        decoded = layers.Dense(input_dimensions / 3, activation=activation, name="decoding_h1")(encoded)
        decoded = layers.Dense(input_dimensions / 2, activation=activation, name="decoding_h2")(decoded)
        decoded = layers.Dense(input_dimensions, name="decoder_output")(decoded)

        # Auto encoder
        ae = keras.Model(input_layers, decoded, name="AE")
        ae.summary()

        # Separate encoder model
        encoder = keras.Model(input_layers, encoded, name="encoder")
        encoder.summary()

        # Separate decoder model
        encoded_input = keras.Input(shape=(embedding_dimensions,))
        deco = ae.layers[-3](encoded_input)
        deco = ae.layers[-2](deco)
        deco = ae.layers[-1](deco)
        # create the decoder model
        decoder = keras.Model(encoded_input, deco, name="decoder")
        decoder.summary()

        # Compile ae
        ae.compile(optimizer="adam", loss=MeanAbsoluteError())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        history = ae.fit(train_data, train_data,
                         epochs=100,
                         batch_size=256,
                         shuffle=True,
                         callbacks=[early_stopping],
                         validation_data=(validation_data, validation_data))

        return ae, encoder, decoder, history
