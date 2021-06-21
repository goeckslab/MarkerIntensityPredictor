from Shared.data import Data
from Shared.data_loader import DataLoader
from services.args_parser import ArgumentParser
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import anndata as ad
import pandas as pd
from pathlib import Path
import umap
import tensorflow as tf
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import phenograph
from sklearn.metrics import r2_score
import seaborn as sns

sns.set_theme(style="darkgrid")


class DenoisingAutoEncoder:
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

    def add_noise(self):
        data = pd.DataFrame(self.normalized_data.X_train.copy())
        means = data.mean()
        std = data.std()

        for column in data:
            random_dist = np.random.normal(means[column], std[column], size=len(data[column]))
            r_column = pd.DataFrame(data[column].sample(frac=0.2))
            noise = np.random.choice(random_dist, size=len(r_column), replace=False)
            r_column[column] = noise

            for index in data.index:
                if index in r_column.index:
                    data[column][index] = r_column[column][index]

        self.normalized_data.X_train_noise = data.to_numpy()

    def build_auto_encoder(self):

        activation = "linear"
        input_layer = keras.Input(shape=(self.inputs_dim,))

        # Encoder
        encoded = layers.Dense(20, activation=activation)(input_layer)
        encoded = layers.Dense(12, activation=activation)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation=activation)(encoded)

        # Decoder
        decoded = layers.Dense(12, activation=activation)(encoded)
        decoded = layers.Dense(20, activation=activation)(decoded)
        decoded = layers.Dense(self.inputs_dim, activation=activation)(decoded)

        # Auto encoder
        self.ae = keras.Model(input_layer, decoded, name="DAE")
        self.ae.summary()

        # Separate encoder model
        self.encoder = keras.Model(input_layer, encoded, name="encoder")
        self.encoder.summary()

        # Separate decoder model
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        deco = self.ae.layers[-3](encoded_input)
        deco = self.ae.layers[-2](deco)
        deco = self.ae.layers[-1](deco)
        # create the decoder model
        self.decoder = keras.Model(encoded_input, deco, name="decoder")
        self.decoder.summary()

        # Compile ae
        self.ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError(), metrics=['acc', 'mean_squared_error'])

        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        self.history = self.ae.fit(self.normalized_data.X_train_noise, self.normalized_data.X_train_noise,
                                   epochs=500,
                                   batch_size=32,
                                   shuffle=True,
                                   callbacks=[callback],
                                   validation_data=(self.normalized_data.X_train, self.normalized_data.X_train))

    def predict(self):
        # Make some predictions
        cell = self.normalized_data.X_val[0]
        cell = cell.reshape(1, cell.shape[0])
        encoded_cell = self.encoder.predict(cell)
        decoded_cell = self.decoder.predict(encoded_cell)
        print(f"Epochs: {len(self.history.history['loss'])}")
        print(f"Input shape:\t{cell.shape}")
        print(f"Encoded shape:\t{encoded_cell.shape}")
        print(f"Decoded shape:\t{decoded_cell.shape}")
        print(f"\nInput:\n{cell[0]}")
        print(f"\nEncoded:\n{encoded_cell[0]}")
        print(f"\nDecoded:\n{decoded_cell[0]}")

    def create_h5ad_object(self):
        # Input
        fit = umap.UMAP()
        self.input_umap = fit.fit_transform(self.normalized_data.X_train_noise)

        # latent space
        fit = umap.UMAP()
        encoded = self.encoder.predict(self.normalized_data.X_train)
        self.latent_umap = fit.fit_transform(encoded)

        self.__create_h5ad(f"latent_markers_{self.encoding_dim}", self.latent_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        self.__create_h5ad(f"input_{self.encoding_dim}", self.input_umap, self.normalized_data.markers,
                           pd.DataFrame(columns=self.normalized_data.markers, data=self.normalized_data.X_train))
        return

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

        plt.title("DAE Scores", y=1.02)
        ax.savefig(Path(f"results/dae/r2_scores_{self.encoding_dim}.png"))
        plt.close()

    def k_means(self):
        # k means determine k
        distortions = []
        K = range(1, 15)
        encoded = self.encoder.predict(self.normalized_data.X_train)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(encoded)
            distortions.append(kmeanModel.inertia_)

        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Optimal k')
        plt.savefig(Path("results", "dae", f"elbow_{self.encoding_dim}.png"))
        plt.close()

    def phenograph(self, encoded_data):
        communities, graph, Q = phenograph.cluster(encoded_data)
        return pd.Series(communities)

    def __create_h5ad(self, file_name: str, umap, markers, df):
        obs = pd.DataFrame(data=df, index=df.index)
        var = pd.DataFrame(index=markers)
        obsm = {"X_umap": umap}
        uns = dict()
        adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns, obsm=obsm)

        adata.write(Path(f'results/dae/{file_name}.h5ad'))
