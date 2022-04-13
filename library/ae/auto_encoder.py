import keras
import mlflow
import pandas as pd
from keras import layers
import tensorflow as tf
from keras.losses import MeanAbsoluteError


class AutoEncoder:

    @staticmethod
    def build_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                           input_dimensions: int, embedding_dimension: int, activation='relu',
                           learning_rate: float = 1e-3,
                           optimizer: str = "adam", use_ml_flow: bool = True, amount_of_layers: int = 3):
        """

        @param training_data:
        @param validation_data:
        @param input_dimensions:
        @param embedding_dimension:
        @param activation:
        @return: Returns the model, encoder, decoder and training history
        """

        if amount_of_layers == 3:
            return AutoEncoder.__build_3_layer_auto_encoder(training_data=training_data,
                                                            validation_data=validation_data,
                                                            input_dimensions=input_dimensions,
                                                            embedding_dimension=embedding_dimension,
                                                            activation=activation, learning_rate=learning_rate,
                                                            optimizer=optimizer, use_ml_flow=use_ml_flow)

        elif amount_of_layers == 5:
            return AutoEncoder.__build_5_layer_auto_encoder(training_data=training_data,
                                                            validation_data=validation_data,
                                                            input_dimensions=input_dimensions,
                                                            embedding_dimension=embedding_dimension,
                                                            activation=activation, learning_rate=learning_rate,
                                                            optimizer=optimizer, use_ml_flow=use_ml_flow)

        else:
            raise ValueError("Only 3 and 5 Layer networks are implemented")

    @staticmethod
    def __build_3_layer_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                     input_dimensions: int,
                                     embedding_dimension: int, activation='relu', learning_rate: float = 1e-3,
                                     optimizer: str = "adam", use_ml_flow: bool = True):
        # activate auto log
        if use_ml_flow:
            mlflow.tensorflow.autolog()

        # Build encoder
        input_layers = keras.Input(shape=(input_dimensions,))
        encoded = layers.Dense(input_dimensions, activation=activation, name="encoding_h1")(input_layers)
        encoded = layers.Dense(input_dimensions / 2, activation=activation, name="encoding_h2")(encoded)
        encoded = layers.Dense(input_dimensions / 3, activation=activation, name="encoding_h3")(encoded)

        # Embedding
        encoded = layers.Dense(embedding_dimension, activation=activation, name="embedding")(encoded)

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
        encoded_input = keras.Input(shape=(embedding_dimension,))
        deco = ae.layers[-3](encoded_input)
        deco = ae.layers[-2](deco)
        deco = ae.layers[-1](deco)
        # create the decoder model
        decoder = keras.Model(encoded_input, deco, name="decoder")
        decoder.summary()

        # Compile ae
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        history = ae.fit(training_data, training_data,
                         epochs=500,
                         batch_size=256,
                         shuffle=True,
                         callbacks=[early_stopping],
                         validation_data=(validation_data, validation_data),
                         verbose=0)

        return ae, encoder, decoder, history

    @staticmethod
    def __build_5_layer_auto_encoder(training_data: pd.DataFrame, validation_data: pd.DataFrame,
                                     input_dimensions: int,
                                     embedding_dimension: int, activation='relu', learning_rate: float = 1e-3,
                                     optimizer: str = "adam", use_ml_flow: bool = True):
        # activate auto log
        if use_ml_flow:
            mlflow.tensorflow.autolog()

        # Build encoder
        input_layers = keras.Input(shape=(input_dimensions,))
        encoded = layers.Dense(input_dimensions, activation=activation, name="encoding_h1")(input_layers)
        encoded = layers.Dense(input_dimensions / 1.5, activation=activation, name="encoding_h2")(encoded)
        encoded = layers.Dense(input_dimensions / 2, activation=activation, name="encoding_h3")(encoded)
        encoded = layers.Dense(input_dimensions / 2.5, activation=activation, name="encoding_h4")(encoded)
        encoded = layers.Dense(input_dimensions / 3, activation=activation, name="encoding_h5")(encoded)

        # Embedding
        encoded = layers.Dense(embedding_dimension, activation=activation, name="embedding")(encoded)

        # Build decoder.
        decoded = layers.Dense(input_dimensions / 3, activation=activation, name="decoding_h1")(encoded)
        decoded = layers.Dense(input_dimensions / 2.5, activation=activation, name="decoding_h2")(decoded)
        decoded = layers.Dense(input_dimensions / 2, activation=activation, name="decoding_h3")(decoded)
        decoded = layers.Dense(input_dimensions / 1.5, activation=activation, name="decoding_h4")(decoded)
        decoded = layers.Dense(input_dimensions, name="decoder_output")(decoded)

        # Auto encoder
        ae = keras.Model(input_layers, decoded, name="AE")
        ae.summary()

        # Separate encoder model
        encoder = keras.Model(input_layers, encoded, name="encoder")
        encoder.summary()

        # Separate decoder model
        encoded_input = keras.Input(shape=(embedding_dimension,))
        deco = ae.layers[-5](encoded_input)
        deco = ae.layers[-4](deco)
        deco = ae.layers[-3](deco)
        deco = ae.layers[-2](deco)
        deco = ae.layers[-1](deco)
        # create the decoder model
        decoder = keras.Model(encoded_input, deco, name="decoder")
        decoder.summary()

        # Compile ae
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        history = ae.fit(training_data, training_data,
                         epochs=500,
                         batch_size=256,
                         shuffle=True,
                         callbacks=[early_stopping],
                         validation_data=(validation_data, validation_data),
                         verbose=0)

        return ae, encoder, decoder, history
