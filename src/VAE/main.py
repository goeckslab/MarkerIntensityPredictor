from VAE.model.vae import VAEModel
import mlflow
from Plotting.plots import Plotting
from pathlib import Path
import pandas as pd
from Shared.data_loader import DataLoader
from Shared.data import Data
from VAE.preprocessing.preprocessing import normalize
from random import randrange
from VAE.evaluation.evaluation import Evaluation


class VAE:
    args = None
    # The vae model
    model = None

    data: Data = None
    # The latent space explorer
    latent_space_explorer = None

    # Contains all evaluation metrics
    evaluation: Evaluation

    # The base results path
    __base_path: Path
    __experiment_id: str

    def __init__(self, args, base_result_path: Path, experiment_id: str):
        self.__base_path = base_result_path
        self.__experiment_id = experiment_id
        self.args = args
        self.__start_experiment()

    def __start_experiment(self):
        # Load cells and markers from the given file
        with mlflow.start_run(run_name="VAE", nested=True, experiment_id=self.__experiment_id) as run:
            mlflow.log_param("file", self.args.file)
            mlflow.log_param("morphological_data", self.args.morph)
            mlflow.set_tag("Model", "VAE")

            # Load data
            cells, markers = DataLoader.load_data(file_name=self.args.file, keep_morph=self.args.morph)
            self.data = Data(cells, markers, normalize)
            self.save_initial_data(cells, markers)

            # Create model
            self.model = VAEModel(self.args, data=self.data, base_results_path=self.__base_path)
            self.model.build_auto_encoder()
            # Log model weights
            self.model.log_model_weights()
            # Create full encode decode dataset
            self.model.encode_decode_test_data()

            self.evaluation = Evaluation(self.__base_path, self.model.vae, self.data)
            self.evaluation.calculate_r2_score()

            plotter = Plotting(self.__base_path, self.args)
            plotter.plot_model_performance(self.model.history, "VAE", "model_performance")
            plotter.plot_reconstructed_markers(X=self.model.data.X_test, X_pred=self.model.reconstructed_data,
                                               markers=self.model.data.markers, sub_directory="Evaluation",
                                               file_name="Initial vs. Reconstructed markers")
            plotter.plot_r2_scores(self.evaluation.r2_scores, "Evaluation", "R^2 Scores")
            plotter.plot_markers(X_train=self.model.data.X_train, X_test=self.model.data.X_test,
                                 X_val=self.model.data.X_val, markers=self.model.data.markers,
                                 sub_directory="Evaluation",
                                 file_name="Marker Expression")

            plotter.plot_weights(self.model.vae.get_layer('encoder').get_layer('encoding_h1').get_weights()[0],
                                 self.data.markers, "Encoding layer")
            plotter.plot_weights(self.model.vae.get_layer('decoder').get_layer('decoder_output').get_weights()[0],
                                 self.data.markers, "Decoding layer")

    def save_initial_data(self, cells, markers):
        cell_save_path = Path(self.__base_path, "cells.csv")
        markers_save_path = Path(self.__base_path, "markers.csv")
        cells.to_csv(cell_save_path, index=False)
        pd.DataFrame(markers).to_csv(markers_save_path, index=False)
        mlflow.log_artifact(str(cell_save_path), "base")
        mlflow.log_artifact(str(markers_save_path), "base")

    def check_mean_and_std(self):
        rnd = randrange(0, self.data.cells.shape[1])
        # Mean should be zero and standard deviation
        # should be 1. However, due to some challenges'
        # regarding floating point positions and rounding,
        # the values should be very close to these numbers.
        # For details, see:
        # https://stackoverflow.com/a/40405912/947889
        # Hence, we assert the rounded values.
        mlflow.log_param("Std", self.data.cells[:, rnd].std())
        mlflow.log_param("Mean", self.data.cells[:, rnd].mean())


if __name__ == "__main__":
    raise "Do not use this file as starting point of the script. Use python3 src/marker_intensity_predictor.py to " \
          "start the script. "
