from AE.ae import AutoEncoderModel
from pathlib import Path
from Shared.data import Data
from Shared.data_loader import DataLoader
import pandas as pd
import mlflow
from AE.preprocessing.preprocessing import normalize
from AE.evaluation.evaluation import Evaluation
from AE.plots.plots import Plotting


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

    def __init__(self, args, base_results_path: Path):
        self.args = args
        self.__base_path = base_results_path
        self.__start_experiment()

    def __start_experiment(self):
        with mlflow.start_run(run_name="AE", nested=True) as run:
            mlflow.log_param("file", self.args.file)
            mlflow.log_param("morphological_data", self.args.morph)

            cells, markers = DataLoader.load_data(file_name=self.args.file, keep_morph=self.args.morph)
            self.data = Data(cells, markers, normalize)

            self.model = AutoEncoderModel(args=self.args, data=self.data, base_result_path=self.__base_path)
            self.model.build_auto_encoder()
            self.model.encode_decode_test_data()

            self.evaluation = Evaluation(self.__base_path, model=self.model.ae, data=self.data)
            self.evaluation.calculate_r2_score()

            Plotting.plot_model_performance(self.model.history, self.__base_path, "Model performance")
            Plotting.plot_reconstructed_markers(self.data.cells, self.model.reconstructed_data, self.data.markers,
                                                self.__base_path, "Input v Reconstructed")
            Plotting.plot_r2_scores(self.evaluation.r2_scores, self.__base_path, "R2 scores")

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
