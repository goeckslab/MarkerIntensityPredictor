import keras
from keras import layers, regularizers
import tensorflow as tf
from entities.data import Data


class AutoEncoder:
    activation = "relu"
    activity_regularizer = regularizers.l1_l2(10e-5)

    # The dimensions of the latent space
    latent_space_dimension: int = 5

    # Encoder part of the network
    encoder = None

    # Decoder part of the network
    decoder = None

    # The combined model
    auto_encoder = None
    # The train history
    train_history = None

    input_layer = None
    output_layer = None

    data: Data = None

    def __init__(self, data: Data):
        self.data = data

    def build_encoder(self):
        self.input_layer = keras.Input(shape=(self.data.inputs_dim,), name=f"input_layers_{self.data.inputs_dim}")

        # Encoder
        encoded = layers.Dense(self.data.inputs_dim / 2, activation=self.activation,
                               activity_regularizer=self.activity_regularizer, name="encoded_2")(self.input_layer)
        encoded = layers.Dense(self.data.inputs_dim / 3, activation=self.activation,
                               activity_regularizer=self.activity_regularizer, name="encoded_3")(encoded)
        # encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.latent_space_dimension, activation=self.activation,
                               activity_regularizer=self.activity_regularizer,
                               name=f"latent_space_{self.latent_space_dimension}")(encoded)
        # Create separate encoder model
        self.encoder = keras.Model(self.input_layer, encoded, name="encoder")

    def build_decoder(self):
        # Separate decoder model
        encoded_input = keras.Input(shape=(self.latent_space_dimension,))
        decoded = layers.Dense(self.data.inputs_dim / 3, activation=self.activation, name="decoded_3")(
            encoded_input)
        decoded = layers.Dense(self.data.inputs_dim / 2, activation=self.activation, name="decoded_2")(
            decoded)
        self.output_layer = layers.Dense(self.data.inputs_dim, activation=self.activation, name="output_layer")(decoded)

        # create the decoder model
        self.decoder = keras.Model(encoded_input, self.output_layer, name="decoder")

    def compile_auto_encoder(self):
        # Auto encoder
        self.auto_encoder = keras.Model(self.input_layer, self.output_layer, name="AE")

    def train_auto_encoder(self):
        # Compile ae
        self.auto_encoder.compile(optimizer="adam", loss=keras.losses.MeanSquaredError(),
                                  metrics=['acc', 'mean_squared_error'])

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.train_history = self.auto_encoder.fit(self.data.X_train, self.data.X_train,
                                                   epochs=500,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   callbacks=[callback],
                                                   validation_data=(self.data.X_val, self.data.X_val))
