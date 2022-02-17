import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from pathlib import Path
from tensorflow import keras
import pandas as pd
import numpy as np
import umap

sns.set_theme()


# Weights
# https://stackoverflow.com/questions/58364974/how-to-load-trained-autoencoder-weights-for-decoder

# Latent space exploration
# https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class LatentSpaceExplorer:
    # The encoded latent space generated by the VAE
    embeddings: pd.DataFrame
    # The list of markers being used
    markers: list
    # The generated cells
    generated_cells: pd.DataFrame
    # Whether plots are created
    __base_results_path = ""

    def __init__(self, embeddings: pd.DataFrame, markers: list, base_results_path: Path):
        self.embeddings = embeddings
        self.markers = markers
        self.__base_results_path = Path(base_results_path)
        self.generated_cells = pd.DataFrame()

        if not self.__base_results_path.exists():
            Path.mkdir(self.__base_results_path)

    def __del__(self):
        if self.__base_results_path.exists():
            Path.unlink(self.__base_results_path)

    def generate_new_cells(self, cells_to_generate: int, not_fixed_dimension: int = 0):
        """
        Explore the latent space
        """
        mlflow.log_param("cells_to_generate", cells_to_generate)
        mlflow.log_param("not_fixed_dimension", not_fixed_dimension)

        if not_fixed_dimension > self.embeddings.shape[1]:
            print("Dimension which should not be fixed is greater than available dimensions. Reset to 0.")
            not_fixed_dimension = 0

        model = keras.models.load_model(
            Path(self.__base_results_path, "VAE", "model", "data", "model"))  # mlflow workaround for the model

        x_values = np.linspace(self.embeddings.min(), self.embeddings.max(), cells_to_generate)
        count: int = 0

        for ix, x in enumerate(x_values):
            # Extract first dimension of latent space

            # Create latent point without fixed dimension
            if not_fixed_dimension == 0:
                latent_point = np.array(x)

            else:
                # Fix dimension
                mean = np.mean(x[[1, 2, 3, 4, 5, 6, 7, 8, 9]])
                first_dim = (x[[0][0]])
                latent_point = np.array([first_dim, mean, mean, mean, mean, mean, mean, mean, mean, mean])

            # input()
            latent_point = latent_point.reshape(1, latent_point.shape[0])
            # Generate new cell
            generated_cell = model.decoder.predict(latent_point)
            self.generated_cells = self.generated_cells.append(pd.Series(generated_cell[0]), ignore_index=True)

            count += 1

        self.generated_cells.columns = self.markers

        save_path = Path(self.__base_results_path, "generated_cells.csv")
        self.generated_cells.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path))

    def plot_generated_cells(self):
        plt.figure(figsize=(20, 9))
        ax = sns.heatmap(self.generated_cells, vmin=self.generated_cells.min().min(),
                         vmax=self.generated_cells.max().max())
        fig = ax.get_figure()
        plt.xlabel("Marker")
        plt.ylabel("Cell")
        plt.tight_layout()

        save_path = Path(self.__base_results_path, "generated_cells.png")
        fig.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close('all')

    def plot_generated_cells_differences(self):
        difference = self.generated_cells.diff(axis=0)
        save_path = Path(self.__base_results_path, "generated_cell_expression_differences.csv")
        difference.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path))

        plt.figure(figsize=(20, 9))
        ax = sns.heatmap(difference, vmin=difference.min().min(), vmax=difference.max().max())
        fig = ax.get_figure()
        plt.xlabel("Difference")
        plt.ylabel("Cell")
        plt.tight_layout()
        save_path = Path(self.__base_results_path, "generated_cell_expression_differences.png")
        fig.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close('all')

    def umap_mapping_of_generated_cells(self):
        fit = umap.UMAP()
        mapping = fit.fit_transform(self.generated_cells)

        plot = sns.scatterplot(data=mapping, x=mapping[:, 0], y=mapping[:, 1],
                               hue=pd.Series(self.generated_cells.index))
        fig = plot.get_figure()
        save_path = Path(self.__base_results_path, "generated_cells_umap.png")
        fig.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close('all')