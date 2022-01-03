import pickle
import sys
from pathlib import Path
from Shared.data import Data
from Shared.data_loader import DataLoader
import numpy as np
import keras
from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import anndata as ad
import pandas as pd
import umap
import tensorflow as tf
from sklearn.metrics import r2_score
import keract as kt
import phenograph
import matplotlib.pyplot as plt
import seaborn as sns


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

    model = None

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

    def load_data(self, data_frame: pd.DataFrame = None):
        print("Loading data...")

        if data_frame is not None:
            inputs, markers = DataLoader.get_data(
                input_dataframe=data_frame, keep_morph=True)
            Data(inputs=np.array(inputs), markers=markers, normalize=self.normalize)

        else:
            if self.args.file:
                inputs, markers = DataLoader.get_data(
                    input_file=self.args.file, keep_morph=True)

            elif self.args.dir:
                inputs, markers = DataLoader.load_folder_data(
                    self.args.dir, True)

            else:
                print("Please specify a directory or a file")
                sys.exit()

        self.data = Data(inputs=np.array(inputs), markers=markers, normalize=self.normalize)

    def build_auto_encoder(self):
        # activation = tf.keras.activations.sigmoid
        activation = tf.keras.activations.relu
        activity_regularizer = regularizers.l1_l2(10e-5)
        input_layer = keras.Input(shape=(self.data.inputs_dim,), name=f"input_layers_{self.data.inputs_dim}")

        # Encoder
        encoded = layers.Dense(self.data.inputs_dim / 2, activation=activation,
                               activity_regularizer=activity_regularizer, name="encoded_2")(input_layer)
        encoded = layers.Dense(self.data.inputs_dim / 3, activation=activation,
                               activity_regularizer=activity_regularizer, name="encoded_3")(encoded)
        # encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation=activation, activity_regularizer=activity_regularizer,
                               name=f"latent_space_{self.encoding_dim}")(
            encoded)
        # encoded = layers.Dropout(0.3)(encoded)

        # Decoder
        decoded = layers.Dense(self.data.inputs_dim / 3, activation=activation, name="decoded_3")(encoded)
        decoded = layers.Dense(self.data.inputs_dim / 2, activation=activation, name="decoded_2")(decoded)
        decoded = layers.Dense(self.data.inputs_dim, activation=activation, name="output_layer")(decoded)

        # Auto encoder
        self.ae = keras.Model(input_layer, decoded, name="AE")
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
        self.ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError(),
                        metrics=['acc', 'mean_squared_error'])

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
        self.input_umap = fit.fit_transform(self.data.X_test)

        # latent space
        fit = umap.UMAP()
        encoded = self.encoder.predict(self.data.X_test)
        self.latent_umap = fit.fit_transform(encoded)

        self.__create_h5ad("latent_markers", self.latent_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_test))
        self.__create_h5ad("input", self.input_umap, self.data.markers,
                           pd.DataFrame(columns=self.data.markers, data=self.data.X_test))
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
        # create phenograph clusters
        communities, graph, Q = phenograph.cluster(self.encoded_data, "leiden", k=30)
        adata = ad.AnnData()
        adata.obs['Phenograph_cluster'] = pd.Categorical(communities)
        adata.obs['Phenograph_Q'] = Q
        fit = umap.UMAP()
        print(self.encoded_data)
        encoded_umap = fit.fit_transform(self.encoded_data)
        scatter = sns.scatterplot(data=encoded_umap, x=encoded_umap[:, 0], y=encoded_umap[:, 1],
                                  c=communities)
        fig = scatter.get_figure()
        plt.show()

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

    def get_activations(self):
        cell = self.data.X_test[0]
        cell = cell.reshape(1, cell.shape[0])
        # activations = kt.get_activations(self.encoder, self.data.X_val)
        activations = kt.get_activations(self.ae, cell)
        print(activations)
        fig = kt.display_activations(activations, cmap="summer", directory=f'{self.results_folder}', save=True)
        kt.display_heatmaps(activations, fig, directory=f'{self.results_folder}', save=True)

    def plot_model(self):
        tf.keras.utils.plot_model(self.ae, to_file=Path(f'{self.results_folder}/model.png', show_shapes=True))
