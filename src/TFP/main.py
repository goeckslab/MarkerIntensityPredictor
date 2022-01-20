import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from args import ArgumentsParser
from typing import Tuple
import pandas as pd
from data_loader import DataLoader
import sys
from entities.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers

# Sub imports
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

datasets, datasets_info = tfds.load(name='mnist',
                                    with_info=True,
                                    as_supervised=False)


def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval.
    image = image < tf.random.uniform(tf.shape(image))  # Randomly binarize.
    return image, image


def preprocess(sample):
    cell = tf.cast(sample, tf.float32)
    cell = sample < tf.random.uniform(tf.shape(cell))  # Randomly binarize.
    return cell, cell


train_dataset = (datasets['train']
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.AUTOTUNE)
                 .shuffle(int(10e3)))
eval_dataset = (datasets['test']
                .map(_preprocess)
                .batch(256)
                .prefetch(tf.data.AUTOTUNE))

print(eval_dataset)
# print(tf.Variable())
input()


def load_data_set(args) -> Tuple[pd.DataFrame, list]:
    """
    Loads the data set given by the cli args
    """
    print("Loading data...")

    inputs: pd.DataFrame
    markers: list

    if args.file:
        inputs, markers = DataLoader.get_data(
            input_file=args.file.name, keep_morph=True)

    else:
        print("Please specify a directory or a file")
        sys.exit()

    return inputs, markers


def normalize(inputs: np.ndarray):
    # Input data contains some zeros which results in NaN (or Inf)
    # values when their log10 is computed. NaN (or Inf) are problematic
    # values for downstream analysis. Therefore, zeros are replaced by
    # a small value; see the following thread for related discussion.
    # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

    inputs[inputs == 0] = 1e-32
    inputs = np.log10(inputs)

    standard_scaler = StandardScaler()
    inputs = standard_scaler.fit_transform(inputs)
    inputs = inputs.clip(min=-5, max=5)

    # min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    # inputs = min_max_scaler.fit_transform(inputs)

    return inputs


class TPFVAE:
    data = None
    inputs = pd.DataFrame()
    markers = []
    vae = None
    train_history = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    encoder = None
    decoder = None

    def __init__(self):
        args = ArgumentsParser.get_args()
        self.inputs, self.markers = load_data_set(args)
        self.data = Data(inputs=np.array(self.inputs), markers=self.markers, normalize=normalize)

        # test = tf.data.Dataset.map(tf.data.Dataset.from_tensor_slices(self.data.X_train).batch(128), preprocess)
        # print(test)
        # input()

        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.data.X_train).batch(128)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(self.data.X_val).batch(128)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.data.X_test).batch(128)

    def create_model(self):
        input_dimensions = self.data.inputs_dim
        latent_space_dimensions = 5
        activation = tf.keras.layers.ReLU()

        # Prior assumption is currently a normal gaussian distribution
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_space_dimensions), scale=1),
                                reinterpreted_batch_ndims=1)

        encoder_inputs = keras.Input(shape=(input_dimensions))
        h1 = layers.Dense(input_dimensions, activation=activation)(encoder_inputs)
        h2 = layers.Dense(input_dimensions / 2, activation=activation)(h1)
        h3 = layers.Dense(input_dimensions / 3, activation=activation)(h2)
        h4 = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_space_dimensions),
                        activation=None)(h3)
        h5 = tfpl.MultivariateNormalTriL(
            latent_space_dimensions,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=1.0))(h4)

        self.encoder = keras.Model(encoder_inputs, h5, name="encoder")

        # Build the decoder
        decoder_inputs = keras.Input(shape=(latent_space_dimensions,))
        h1 = layers.Dense(input_dimensions / 3, activation=activation)(decoder_inputs)
        h2 = layers.Dense(input_dimensions / 2, activation=activation)(h1)

        decoder_outputs = layers.Dense(input_dimensions)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

        print(self.encoder.summary())
        print(self.decoder.summary())
        self.vae = tfk.Model(inputs=self.encoder.inputs,
                             outputs=self.decoder(self.encoder.outputs[0]))

    def train_model(self):
        negloglik = lambda x, rv_x: -rv_x.log_prob(x)

        self.vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3)
                         , loss=negloglik)

        callback = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)

        _ = self.vae.fit(self.train_dataset,
                         epochs=15,
                         validation_data=eval_dataset,
                         batch_size=32,
                         verbose=1)

        # self.train_history = self.vae.fit(self.data.X_train,
        #                                 epochs=15,
        #                                 validation_data=(self.data.X_val, self.data.X_val),
        #                                 callbacks=callback,
        #                                 batch_size=96,
        #                                 shuffle=True,
        #                                 verbose=1
        #                                 )


vae = TPFVAE()
vae.create_model()
vae.train_model()
