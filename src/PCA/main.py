import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Shared.data_loader import DataLoader
import sys
from Shared.data import Data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class PCAMode:
    data: Data
    args = None
    results_folder = Path("results", "pca")

    def __init__(self, args):
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

    def reduce_dimensions(self):
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(self.data.X_test)
        # Plotting the variances for each PC
        components = range(1, pca.n_components_ + 1)
        fig = plt.figure(figsize=(10, 5))

        plt.bar(components, pca.explained_variance_ratio_, color='black')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance %')
        plt.xticks(components)
        fig.savefig(f"{self.results_folder}/pca_components.png")

        # Putting components in a dataframe for later
        pca_components = pd.DataFrame(principal_components)

        inertias = []

        # Creating 10 K-Mean models while varying the number of clusters (k)
        for k in range(1, 10):
            model = KMeans(n_clusters=k)

            # Fit model to samples
            model.fit(pca_components.iloc[:, :3])

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        fig = plt.figure(figsize=(10, 5))

        plt.plot(range(1, 10), inertias, '-p', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        # plt.xticks(ks)
        fig.figure.savefig(f"{self.results_folder}/k_means.png")

        model = KMeans(n_clusters=3)
        model.fit(pca_components.iloc[:, :2])

        fig = plt.figure(figsize=(10, 5))
        labels = model.predict(pca_components.iloc[:, :2])
        plt.scatter(pca_components[0], pca_components[1], c=labels)
        fig.savefig(f"{self.results_folder}/pca_clusters.png")
