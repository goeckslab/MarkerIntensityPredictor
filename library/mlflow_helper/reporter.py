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
        if prefix is not None:
            save_path = Path(save_path, f"{prefix}_r2_score.csv")
        else:
            save_path = Path(save_path, "r2_score.csv")
        r2_scores.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path), mlflow_folder)
