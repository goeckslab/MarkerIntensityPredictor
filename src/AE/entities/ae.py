from entities.data import Data
from services.data_loader import DataLoader
from services.args_parser import ArgumentParser
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from services.plots import Plots
import anndata as ad
import pandas as pd
from pathlib import Path
import umap
import tensorflow as tf
import sys
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import phenograph

sns.set_theme(style="darkgrid")


class AutoEncoder:
    un_normalized_data: Data
    normalized_data: Data

    # The defined encoder
    encoder: any
    # The defined decoder
    decoder: any
    # The ae
    ae: any

    history: any

    input_dim: int
    encoding_dim: int

    input_umap: any
    latent_umap: any

    r2_scores = pd.DataFrame(columns=["Marker", "Score"])

    def __init__(self):
        self.encoding_dim = 5

    def load_data(self):
        print("Loading data...")
        self.un_normalized_data = Data()
        args = ArgumentParser.get_args()

        if args.file:
            inputs, self.un_normalized_data.markers = DataLoader.get_data(
                ArgumentParser.get_args().file)

        elif args.dir:
            inputs, self.un_normalized_data.markers = DataLoader.load_folder_data(
                args.dir)

        else:
            print("Please specify a directory or a file")
            sys.exit()

        self.un_normalized_data.inputs = np.array(inputs)

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

    def split_data(self):
        print("Splitting data")
        X_dev, X_val = train_test_split(self.un_normalized_data.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

        self.un_normalized_data.X_train = X_train
        self.un_normalized_data.X_test = X_test
        self.un_normalized_data.X_val = X_val

        # Store the normalized data
        self.normalized_data = Data()
        self.normalized_data.markers = self.un_normalized_data.markers
        self.normalized_data.inputs = np.array(self.normalize(self.un_normalized_data.inputs))
        self.normalized_data.X_train = self.normalize(X_train)
        self.normalized_data.X_test = self.normalize(X_test)
        self.normalized_data.X_val = self.normalize(X_val)

        self.inputs_dim = self.normalized_data.inputs.shape[1]

    def build_auto_encoder(self):
        activation = 'linear'
        # This is our input image
        encoder_input = keras.Input(shape=(self.inputs_dim,))
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim, activation=activation)(encoder_input)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.inputs_dim, activation=activation)(encoded)

        # This model maps an input to its reconstruction
        self.ae = keras.Model(encoder_input, decoded)

        print(self.ae.summary())

        self.encoder = keras.Model(encoder_input, encoded)

        # This is our encoded (32-dimensional) input
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        # Retrieve the last layer of the auto encoder model
        decoder_layer = self.ae.layers[-1]
        # Create the decoder model
        self.decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

        self.ae.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(),  metrics=['acc', 'mean_squared_error'])

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.history = self.ae.fit(self.normalized_data.X_train, self.normalized_data.X_train,
                                   epochs=500,
                                   batch_size=32,
                                   shuffle=True,
                                   callbacks=[callback],
                                   validation_data=(self.normalized_data.X_test, self.normalized_data.X_test))

    def predict(self):
        # Make some predictions
        cell = self.normalized_data.X_val[0]
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
        recon_val = self.ae.predict(self.normalized_data.X_val)

        recon_val = pd.DataFrame(data=recon_val, columns=self.normalized_data.markers)
        input_data = pd.DataFrame(data=self.normalized_data.X_val, columns=self.normalized_data.markers)

        for marker in self.normalized_data.markers:
            input_marker = input_data[f"{marker}"]
            var_marker = recon_val[f"{marker}"]

            score = r2_score(input_marker, var_marker)
            self.r2_scores = self.r2_scores.append(
                {
                    "Marker": marker,
                    "Score": score
                }, ignore_index=True
            )

        # Plot it
        ax = sns.catplot(
            data=self.r2_scores, kind="bar",
            x="Score", y="Marker", ci="sd", palette="dark", alpha=.6, height=6
        )
        ax.despine(left=True)
        ax.set_axis_labels("R2 Score", "Marker")
        ax.set(xlim=(0, 1))

        plt.title("AE Scores", y=1.02)
        ax.savefig(Path(f"results/ae/r2_scores_{self.encoding_dim}.png"))
        plt.close()

    def create_h5ad_object(self):
        # Input
        fit = umap.UMAP()
        self.input_umap = input_umap = fit.fit_transform(self.normalized_data.X_train)

        # latent space
        fit = umap.UMAP()
        encoded = self.encoder.predict(self.normalized_data.X_train)
        self.latent_umap = fit.fit_transform(encoded)

        self.__create_h5ad("latent_markers", self.latent_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        self.__create_h5ad("input", input_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        return

    def phenograph(self, encoded_data):
        communities, graph, Q = phenograph.cluster(encoded_data)
        return pd.Series(communities)

    def plots(self):
        clusters = self.phenograph(self.encoder.predict(self.normalized_data.X_train))
        Plots.plot_model_performance(self.history, f"model_performance_{self.encoding_dim}")
        Plots.plot_reconstructed_validation_markers(self.ae, self.normalized_data.X_val, self.normalized_data.markers,
                                                    f"reconstructed_intensities_{self.encoding_dim}")
        Plots.latent_space_cluster(self.input_umap, self.latent_umap, clusters,
                                   f"latent_space_clusters_{self.encoding_dim}")

    def __create_h5ad(self, file_name: str, umap, markers, df):
        obs = pd.DataFrame(data=df, index=df.index)
        var = pd.DataFrame(index=markers)
        obsm = {"X_umap": umap}
        uns = dict()
        adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns, obsm=obsm)

        adata.write(Path(f'results/ae/{file_name}.h5ad'))