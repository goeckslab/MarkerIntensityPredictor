import pickle
import sys
from pathlib import Path
from Shared.data import Data
from Shared.data_loader import DataLoader
import numpy as np
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import anndata as ad
import pandas as pd
import umap
import tensorflow as tf
from sklearn.metrics import r2_score


class AutoEncoder:
    data: Data

    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    ae: any

    # the training history of the AE
    history: any

    input_dim: int
    encoding_dim: int

    input_umap: any
    latent_umap: any

    r2_scores = pd.DataFrame(columns=["Marker", "Score"])
    encoded_data = pd.DataFrame()
    reconstructed_data = pd.DataFrame()
    args = None
    results_folder = Path("results", "ae")

    def __init__(self, args):
        self.encoding_dim = 5
        self.args = args

    def normalize(self, data):
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        data[data == 0] = 1e-32
        data = np.log10(data)

        standard_scaler = StandardScaler()
        data = standard_scaler.fit_transform(data)
        data = data.clip(min=-5, max=5)

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        data = min_max_scaler.fit_transform(data)
        return data

    def load_data(self):
        print("Loading data...")

        if self.args.file:
            inputs, markers = DataLoader.get_data(
                self.args.file)

        elif self.args.dir:
            inputs, markers = DataLoader.load_folder_data(
                self.args.dir)

        else:
            print("Please specify a directory or a file")
            sys.exit()

        self.data = Data(np.array(inputs), markers, self.normalize)

    def build_auto_encoder(self):
        activation = 'linear'
        # This is our input image
        encoder_input = keras.Input(shape=(self.data.inputs_dim,))
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim, activation=activation)(encoder_input)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.data.inputs_dim, activation=activation)(encoded)

        # This model maps an input to its reconstruction
        self.ae = keras.Model(encoder_input, decoded)

        print(self.ae.summary())

        self.encoder = keras.Model(encoder_input, encoded)

        # This is our encoded (21-dimensional) input
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        # Retrieve the last layer of the auto encoder model
        decoder_layer = self.ae.layers[-1]
        # Create the decoder model
        self.decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

        self.ae.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['acc', 'mean_squared_error'])

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.history = self.ae.fit(self.data.X_train, self.data.X_train,
                                   epochs=500,
                                   batch_size=32,
                                   shuffle=True,
                                   callbacks=[callback],
                                   validation_data=(self.data.X_val, self.data.X_val))

    def predict(self):
        # Make some predictions
        cell = self.data.X_test[0]
        cell = cell.reshape(1, cell.shape[0])
        encoded_cell = self.encoder.predict(cell)
        decoded_cell = self.decoder.predict(encoded_cell)
        # var_cell = self.ae.predict(cell)
        print(f"Epochs: {len(self.history.history['loss'])}")
        print(f"Input shape:\t{cell.shape}")
        print(f"Encoded shape:\t{encoded_cell.shape}")
        print(f"Decoded shape:\t{decoded_cell.shape}")
        print(f"\nInput:\n{cell[0]}")
        print(f"\nEncoded:\n{encoded_cell[0]}")
        print(f"\nDecoded:\n{decoded_cell[0]}")

    def calculate_r2_score(self):
        recon_test = self.ae.predict(self.data.X_test)
        recon_test = pd.DataFrame(data=recon_test, columns=self.data.markers)
        input_data = pd.DataFrame(data=self.data.X_test, columns=self.data.markers)

        for marker in self.data.markers:
            input_marker = input_data[f"{marker}"]
            var_marker = recon_test[f"{marker}"]

            score = r2_score(input_marker, var_marker)
            self.r2_scores = self.r2_scores.append(
                {
                    "Marker": marker,
                    "Score": score
                }, ignore_index=True
            )

    def create_h5ad_object(self):
        # Input
        fit = umap.UMAP()
        self.input_umap = input_umap = fit.fit_transform(self.data.X_train)

        # latent space
        fit = umap.UMAP()
        encoded = self.encoder.predict(self.data.X_train)
        self.latent_umap = fit.fit_transform(encoded)

        self.__create_h5ad("latent_markers", self.latent_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_train))
        self.__create_h5ad("input", input_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_train))
        return

    def __create_h5ad(self, file_name: str, umap, markers, df):
        obs = pd.DataFrame(data=df, index=df.index)
        var = pd.DataFrame(index=markers)
        obsm = {"X_umap": umap}
        uns = dict()
        adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns, obsm=obsm)

        adata.write(Path(f'{self.results_folder}/{file_name}.h5ad'))

    def create_test_predictions(self):
        self.encoded_data = pd.DataFrame(self.encoder.predict(self.data.X_test))
        self.reconstructed_data = pd.DataFrame(columns=self.data.markers, data=self.decoder.predict(self.encoded_data))

    def create_correlation_data(self):
        inputs = pd.DataFrame(columns=self.data.markers, data=self.data.inputs)
        corr = inputs.corr()
        corr.to_csv(Path(f'{self.results_folder}/correlation.csv'), index=False)

    def write_created_data_to_disk(self):
        with open(f'{self.results_folder}/ae_history', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        X_test = pd.DataFrame(columns=self.data.markers, data=self.data.X_test)

        X_test.to_csv(Path(f'{self.results_folder}/test_data.csv'), index=False)
        self.encoded_data.to_csv(Path(f'{self.results_folder}/encoded_data.csv'), index=False)
        self.reconstructed_data.to_csv(Path(f'{self.results_folder}/reconstructed_data.csv'), index=False)
        self.r2_scores.to_csv(Path(f'{self.results_folder}/r2_scores.csv'), index=False)
