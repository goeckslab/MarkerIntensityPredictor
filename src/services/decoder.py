import keras
from keras import layers

class Decoder:
    @staticmethod
    def build_decoder(latent_dimensions: int, nbl:int, inputs_dim, activation):
        # Build the decoder
        decoder_inputs = keras.Input(shape=(latent_dimensions,))
        h1 = layers.Dense(nbl, activation=activation)(decoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation)(h1)

        decoder_outputs = layers.Dense(inputs_dim)(h2)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()
