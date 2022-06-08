from pathlib import Path
import mlflow
import pandas as pd


class Reporter:
    @staticmethod
    def report_cells_and_features(save_path: Path, cells, features: list, prefix: str = None,
                                  mlflow_folder: str = None):

        if prefix is None:
            cell_save_path = Path(save_path, "cells.csv")
            features_save_path = Path(save_path, "features.csv")
        else:
            cell_save_path = Path(save_path, f"{prefix}_cells.csv")
            features_save_path = Path(save_path, f"{prefix}_features.csv")

        cells.to_csv(cell_save_path, index=False)
        pd.DataFrame(features).to_csv(features_save_path, index=False)
        mlflow.log_artifact(str(cell_save_path), "base")
        mlflow.log_artifact(str(features_save_path), "base")

    @staticmethod
    def report_r2_scores(r2_scores: pd.DataFrame, save_path: Path, mlflow_folder: str = None, prefix: str = None,
                         use_mlflow: bool = True):
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

        if use_mlflow:
            if mlflow_folder is None:
                mlflow.log_artifact(str(save_path))
            else:
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

    @staticmethod
    def report_evaluation(evaluations: list, file_name: str, save_path: Path, mlflow_folder: str):
        df = pd.DataFrame(evaluations)
        save_path = Path(save_path, f"{file_name}.csv")
        df.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path), mlflow_folder)

    @staticmethod
    def upload_csv(data: pd.DataFrame, save_path: Path, file_name: str, mlflow_folder: str = None):
        save_path = Path(save_path, f"{file_name}.csv")
        data.to_csv(save_path, index=False)
        if mlflow_folder is not None:
            mlflow.log_artifact(str(save_path), mlflow_folder)
        else:
            mlflow.log_artifact(str(save_path))
