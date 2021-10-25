from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Shared.data import Data
import numpy as np
from random import randrange
import pandas as pd
from pathlib import Path
import keras
from VAE.sampling import Sampling
from VAE.vae_model import VAE
import anndata as ad
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score
import umap
import tensorflow as tf
import keract as kt

# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class VAutoEncoder:
    # The whole data set
    data_set = pd.DataFrame()
    # Markers contained in the data set
    markers = pd.Series()
    # The split data
    data: Data

    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    vae: any

    # the training history of the AE
    history: any

    # the input dimensions
    input_dim: int

    # the latent space dimensions
    encoding_dim: int

    # the umap representation of the input
    input_umap: any

    # the umap representation of the latent space
    latent_umap: any

    r2_scores = pd.DataFrame(columns=["Marker", "Score"])
    encoded_data = pd.DataFrame()
    reconstructed_data = pd.DataFrame()

    args = None
    # The default results folder, where all experiment data is being stored. Can be overriden by the constructor
    results_folder = Path("results", "vae")

    def __init__(self, args, dataset: pd.DataFrame, markers: list, results_folder: Path = None,
                 train: pd.DataFrame = None, test: pd.DataFrame = None):
        self.encoding_dim = 5
        self.args = args
        self.markers = markers
        self.data_set = dataset

        if results_folder is not None:
            self.results_folder = results_folder

        # Generate data object with given data
        if train is not None and test is not None:
            self.data = Data(train=train, test=test, markers=self.markers, normalize=self.normalize)
        else:
            self.data = Data(inputs=np.array(self.data_set), markers=self.markers, normalize=self.normalize)

    def normalize(self, inputs: np.ndarray):
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

    def check_mean_and_std(self):
        rnd = randrange(0, self.data.inputs.shape[1])
        # Mean should be zero and standard deviation
        # should be 1. However, due to some challenges
        # relationg to floating point positions and rounding,
        # the values should be very close to these numbers.
        # For details, see:
        # https://stackoverflow.com/a/40405912/947889
        # Hence, we assert the rounded values.
        print(self.data.inputs)
        print(self.data.inputs[:, rnd])

        print(f"Std: {self.data.inputs[:, rnd].std()}")
        print(f"Mean: {self.data.inputs[:, rnd].mean()}")

    def build_auto_encoder(self):
        # Build the encoder

        inputs_dim = self.data.inputs_dim
        r = regularizers.l1_l2(10e-5)
        activation = tf.keras.layers.ReLU()

        # encoder_inputs = keras.Input(shape=(inputs_dim,))
        # h1 = layers.Dense(inputs_dim, activation=activation, activity_regularizer=activity_regularizer)(encoder_inputs)
        # h2 = layers.BatchNormalization()(h1)
        # h3 = layers.Dropout(0.2)(h2)
        # h4 = layers.Dense(inputs_dim / 2, activation=activation, activity_regularizer=activity_regularizer)(h3)
        # h5 = layers.BatchNormalization()(h4)
        # h6 = layers.Dropout(0.2)(h5)
        # h7 = layers.Dense(inputs_dim / 3, activation=activation, activity_regularizer=activity_regularizer)(h6)
        # h8 = layers.Dropout(0.2)(h7)
        # h9 = layers.BatchNormalization()(h8)

        # The following variables are for the convenience of building the decoder.
        # last layer before flatten
        # lbf = h9
        # shape before flatten.
        # sbf = keras.backend.int_shape(lbf)[1:]
        # neurons count before latent dim
        # nbl = np.prod(sbf)

        encoder_inputs = keras.Input(shape=(inputs_dim))
        h1 = layers.Dense(inputs_dim, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(inputs_dim / 3, activation=activation, activity_regularizer=r)(h2)

        # The following variables are for the convenience of building the decoder.
        # last layer before flatten
        lbf = h3
        # shape before flatten.
        sbf = keras.backend.int_shape(lbf)[1:]
        # neurons count before latent dim
        nbl = np.prod(sbf)

        z_mean = layers.Dense(self.encoding_dim, name="z_mean")(lbf)
        z_log_var = layers.Dense(self.encoding_dim, name="z_log_var")(lbf)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(self.encoding_dim,))
        h1 = layers.Dense(nbl, activation=activation)(decoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation)(h1)

        decoder_outputs = layers.Dense(inputs_dim)(h2)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

        # Visualize the model.
        # tf.keras.utils.plot_model(model, to_file="model.png")

        # Train the VAE
        # Create the VAR, compile, and run.

        callback = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005))

        self.history = self.vae.fit(self.data.X_train,
                                    validation_data=(self.data.X_val, self.data.X_val),
                                    epochs=500,
                                    callbacks=callback,
                                    batch_size=32,
                                    shuffle=True,
                                    verbose=1)

    def predict(self):
        # Make some predictions
        cell = self.data.X_test[0]
        cell = cell.reshape(1, cell.shape[0])
        mean, log_var, z = self.encoder.predict(cell)
        encoded_cell = z
        decoded_cell = self.decoder.predict(encoded_cell)
        var_cell = self.vae.predict(cell)
        print(f"Input shape:\t{cell.shape}")
        print(f"Encoded shape:\t{encoded_cell.shape}")
        print(f"Decoded shape:\t{decoded_cell.shape}")
        print(f"\nInput:\n{cell[0]}")
        print(f"\nEncoded:\n{encoded_cell[0]}")
        print(f"\nDecoded:\n{decoded_cell[0]}")

    def calculate_r2_score(self):
        recon_test = self.vae.predict(self.data.X_test)
        recon_test = pd.DataFrame(data=recon_test, columns=self.data.markers)
        input_data = pd.DataFrame(data=self.data.X_test, columns=self.data.markers)

        # self.plot_clusters(input_data, range(len(self.data.markers)))

        for marker in self.data.markers:
            input_marker = input_data[f"{marker}"]
            var_marker = recon_test[f"{marker}"]

            self.r2_scores = self.r2_scores.append(
                {
                    "Marker": marker,
                    "Score": r2_score(input_marker, var_marker)
                }, ignore_index=True
            )
        # self.plot_label_clusters(self.data.X_test, self.data.X_test)

    def plot_label_clusters(self, data, labels):
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.vae.encoder.predict(data)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

    def create_h5ad_object(self):
        # Input
        fit = umap.UMAP()
        self.input_umap = input_umap = fit.fit_transform(self.data.X_test)

        # latent space
        fit = umap.UMAP()
        mean, log_var, z = self.encoder.predict(self.data.X_test)
        self.latent_umap = fit.fit_transform(z)

        self.__create_h5ad("latent_markers", self.latent_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_test))
        self.__create_h5ad("input", input_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_test))
        return

    def __create_h5ad(self, file_name: str, umap, markers, df):
        obs = pd.DataFrame(data=df, index=df.index)
        var = pd.DataFrame(index=markers)
        obsm = {"X_umap": umap}
        uns = dict()
        adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns, obsm=obsm)

        adata.var_names_make_unique()
        adata.write(Path(f'{self.results_folder}/{file_name}.h5ad'))

    def create_test_predictions(self):
        mean, log_var, z = self.encoder.predict(self.data.X_test)
        self.encoded_data = pd.DataFrame(z)
        self.reconstructed_data = pd.DataFrame(columns=self.data.markers, data=self.decoder.predict(self.encoded_data))

    def create_correlation_data(self):
        inputs = pd.DataFrame(columns=self.data.markers, data=self.data_set)
        corr = inputs.corr()
        corr.to_csv(Path(f'{self.results_folder}/correlation.csv'), index=False)

    def write_created_data_to_disk(self):
        with open(f'{self.results_folder}/vae_history', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        self.vae.save(f"{self.results_folder}/model")

        X_test = pd.DataFrame(columns=self.data.markers, data=self.data.X_test)

        pd.DataFrame(columns=["names"], data=self.data.markers).to_csv(Path(f'{self.results_folder}/markers.csv'),
                                                                       index=False)
        X_test.to_csv(Path(f'{self.results_folder}/test_data.csv'), index=False)
        self.encoded_data.to_csv(Path(f'{self.results_folder}/vae_encoded_data.csv'), index=False)
        self.reconstructed_data.to_csv(Path(f'{self.results_folder}/reconstructed_data.csv'), index=False)
        self.r2_scores.to_csv(Path(f'{self.results_folder}/r2_scores.csv'), index=False)

    def get_activations(self):
        cell = self.data.X_test[0]
        cell = cell.reshape(1, cell.shape[0])
        # activations = kt.get_activations(self.encoder, self.data.X_val)
        activations = kt.get_activations(self.vae, cell)
        fig = kt.display_activations(activations, cmap="summer", directory=f'{self.results_folder}', save=True)
