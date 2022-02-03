from VAE.model.vae import VAEModel
import logging
import mlflow
from VAE.plots.plots import Plotting
from VAE.latentspace.laten_space_exploration import LatentSpaceExplorer
from pathlib import Path
import pandas as pd
from Shared.data_loader import DataLoader

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class VAE:
    args = None
    # The vae model
    model = None

    # The latent space explorer
    latent_space_explorer = None

    __base_path: Path

    def __init__(self, args, base_result_path: Path):
        self.__base_path = base_result_path
        self.args = args
        self.__start_experiment()

    def __start_experiment(self):
        # Load cells and markers from the given file
        with mlflow.start_run(run_name="VAE", nested=True) as run:
            mlflow.log_param("file", self.args.file)
            mlflow.log_param("morphological_data", self.args.morph)
            cells, markers = DataLoader.load_data(file_name=self.args.file, keep_morph=self.args.morph)
            self.save_initial_data(cells, markers)
            model = VAEModel(self.args, cells, markers, self.__base_path)
            model.build_auto_encoder()
            model.encode_decode_test_data()
            model.calculate_r2_score()
            Plotting.plot_model_performance(model.history, self.__base_path, "model_performance")
            Plotting.plot_reconstructed_markers(model.data.X_test, model.reconstructed_data, model.data.markers,
                                                self.__base_path,
                                                "Initial vs. Reconstructed markers")
            Plotting.plot_r2_scores(model.r2_scores, self.__base_path, "R^2 Scores")

            latent_space_explorer = LatentSpaceExplorer(model.encoded_data, model.data.markers, self.__base_path)
            latent_space_explorer.explore_latent_space(latent_space_dimensions=model.latent_space_dimensions,
                                                       cells_to_generate=4000)

    def save_initial_data(self, cells, markers):
        cell_save_path = Path(self.__base_path, "cells.csv")
        markers_save_path = Path(self.__base_path, "markers.csv")
        cells.to_csv(cell_save_path, index=False)
        pd.DataFrame(markers).to_csv(markers_save_path, index=False)
        mlflow.log_artifact(str(cell_save_path), "base")
        mlflow.log_artifact(str(markers_save_path), "base")


if __name__ == "__main__":
    raise "Do not use this file as starting point of the script. Use python3 src/marker_intensity_predictor.py to " \
          "start the script. "
