from pathlib import Path
import mlflow
import pandas as pd


class Reporter:
    @staticmethod
    def report_cells_and_markers(save_path: Path, cells, markers: list):
        cell_save_path = Path(save_path, "cells.csv")
        markers_save_path = Path(save_path, "markers.csv")
        cells.to_csv(cell_save_path, index=False)
        pd.DataFrame(markers).to_csv(markers_save_path, index=False)
        mlflow.log_artifact(str(cell_save_path), "base")
        mlflow.log_artifact(str(markers_save_path), "base")
