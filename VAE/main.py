from src.data.folder_management import FolderManagement
from src.arguments.args_parser import ArgumentParser
from src.model.main import VAutoEncoder
from src.data.data_loader import DataLoader
import logging
import mlflow
from src.plots.plots import Plotting
from src.latentspace.laten_space_exploration import LatentSpaceExplorer
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def save_initial_data(cells, markers):
    cell_save_path = Path("VAE", "results", "cells.csv")
    markers_save_path = Path("VAE", "results", "markers.csv")
    cells.to_csv(cell_save_path, index=False)
    pd.DataFrame(markers).to_csv(markers_save_path, index=False)
    mlflow.log_artifact(cell_save_path)
    mlflow.log_artifact(markers_save_path)


def handle_model_training(args):
    # Load cells and markers from the given file
    with mlflow.start_run(run_name=args.name) as run:
        mlflow.log_param("file", args.file)
        mlflow.log_param("morphological_data", args.morph)
        cells, markers = DataLoader.load_data(file_name=args.file)
        save_initial_data(cells, markers)
        vae = VAutoEncoder(args, cells, markers)
        vae.build_auto_encoder()
        vae.encode_decode_test_data()
        vae.calculate_r2_score()
        Plotting.plot_model_performance(vae.history, "model_performance")
        Plotting.plot_reconstructed_markers(vae.data.X_test, vae.reconstructed_data, vae.data.markers,
                                            "Initial vs. Reconstructed markers")
        Plotting.plot_r2_scores(vae.r2_scores, "R^2 Scores")

        latent_space_explorer = LatentSpaceExplorer(vae.encoded_data, vae.data.markers)
        latent_space_explorer.explore_latent_space(latent_space_dimensions=vae.latent_space_dimensions,
                                                   cells_to_generate=4000)


if __name__ == "__main__":
    args = ArgumentParser.get_args()
    FolderManagement.create_folders(args)

    # Get invoked parser
    invoked_parser = args.command

    if invoked_parser == "model":
        handle_model_training(args=args)
