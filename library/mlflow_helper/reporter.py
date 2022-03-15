from pathlib import Path
import mlflow
import pandas as pd


class Reporter:
    @staticmethod
    def report_cells_and_markers(save_path: Path, cells, markers: list, prefix: str = None):

        if prefix is None:
            cell_save_path = Path(save_path, "cells.csv")
            markers_save_path = Path(save_path, "markers.csv")
        else:
            cell_save_path = Path(save_path, f"{prefix}_cells.csv")
            markers_save_path = Path(save_path, f"{prefix}_markers.csv")

        cells.to_csv(cell_save_path, index=False)
        pd.DataFrame(markers).to_csv(markers_save_path, index=False)
        mlflow.log_artifact(str(cell_save_path), "base")
        mlflow.log_artifact(str(markers_save_path), "base")

    @staticmethod
    def report_r2_scores(r2_scores: pd.DataFrame, save_path: Path, mlflow_folder: str, prefix: str = None):
        """
        Report r2 scores
        @param r2_scores:
        @param save_path:
        @param mlflow_folder:
        @param prefix:
        @return:
        """
        if prefix is not None:
            save_path = Path(save_path, f"{prefix}_r2_score.csv")
        else:
            save_path = Path(save_path, "r2_score.csv")
        r2_scores.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path), mlflow_folder)

    @staticmethod
    def report_weights(weights, markers: list, save_path: Path, mlflow_folder: str, file_name: str):
        """
        Uploads the given weights artifact to mlflow
        @param weights:
        @param markers:
        @param save_path:
        @param mlflow_folder:
        @param file_name:
        @return:
        """
        df = pd.DataFrame(weights, columns=markers)
        save_path = Path(save_path, f"{file_name}.csv")
        df.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path), mlflow_folder)

    @staticmethod
    def report_r2_score_mean_difference(r2score_difference: pd.DataFrame, save_path: Path, prefix: str = None):
        if prefix is not None:
            save_path = Path(save_path, f"{prefix}_r2_score_mean_difference.csv")
        else:
            save_path = Path(save_path, f"r2_score_mean_difference.csv")
        r2score_difference.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path))
