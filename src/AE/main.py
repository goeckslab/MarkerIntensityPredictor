from AE.ae import AutoEncoderModel
from pathlib import Path
from Shared.data import Data
from Shared.data_loader import DataLoader
import pandas as pd
import mlflow
from AE.preprocessing.preprocessing import normalize
from AE.evaluation.evaluation import Evaluation
from Plotting.plots import Plotting
from random import randrange
from ClusterAnalysis.main import ClusterAnalysis


class AutoEncoder:
    # The arguments passed from the cli
    args = None

    # the base path
    __base_path: Path

    # The data being used to train and evaluate the model
    data: Data

    # The auto encoder model
    model = None

    # Contains all evaluation metrics
    evaluation: Evaluation

    # The associated experiment id
    __experiment_id: str

    def __init__(self, args, base_results_path: Path, experiment_id: str):
        self.args = args
        self.__base_path = base_results_path
        self.__experiment_id = experiment_id
        self.__start_experiment()

    def __start_experiment(self):
        with mlflow.start_run(run_name="AE", nested=True, experiment_id=self.__experiment_id) as run:
            mlflow.log_param("file", self.args.file)
            mlflow.log_param("morphological_data", self.args.morph)
            mlflow.set_tag("Group", self.args.group)
            mlflow.set_tag("Model", "AE")

            # Load data
            cells, markers = DataLoader.load_data(file_name=self.args.file, keep_morph=self.args.morph)
            self.data = Data(cells, markers, normalize)
            self.save_initial_data(cells, markers)

            # Create model
            self.model = AutoEncoderModel(args=self.args, data=self.data, base_result_path=self.__base_path)
            self.model.build_auto_encoder()
            self.model.encode_decode_test_data()

            self.evaluation = Evaluation(self.__base_path, model=self.model.ae, data=self.data)
            self.evaluation.calculate_r2_score()

            plotter = Plotting(self.__base_path)
            plotter.plot_model_performance(self.model.history, "AE", "Model performance")
            plotter.plot_reconstructed_markers(self.data.cells, self.model.reconstructed_data, self.data.markers,
                                               "Evaluation", "Input v Reconstructed")
            plotter.plot_r2_scores(self.evaluation.r2_scores, "Evaluation", "R2 scores")
            plotter.plot_markers(X_train=self.data.X_train, X_test=self.data.X_test,
                                 X_val=self.data.X_val, markers=self.data.markers,
                                 sub_directory="Evaluation",
                                 file_name="Marker Expression")

            cluster_analyser = ClusterAnalysis(self.__base_path)
            cluster_analyser.create_umap_cluster(self.model.encoded_data, "Cluster Analysis")

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
