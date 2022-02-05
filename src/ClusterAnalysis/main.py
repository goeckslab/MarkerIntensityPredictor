from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import umap


class ClusterAnalysis:
    __base_results_folder: Path

    def __init__(self, base_results_folder: Path):
        self.__base_results_folder = base_results_folder

    def create_umap_cluster(self, encoded_data: pd.DataFrame, sub_directory: str):
        clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(encoded_data)

        plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=0.1, cmap='Spectral')
        save_path = Path(self.__base_results_folder, "umap_cluster.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
