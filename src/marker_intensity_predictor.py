from args import ArgumentParser
from Shared.folder_management import FolderManagement
from Shared.data_loader import DataLoader
import mlflow
from VAE.main import VAE
from AE.main import AutoEncoder
from pathlib import Path

vae_base_result_path = Path("results", "VAE")
ae_base_result_path = Path("results", "AE")

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    FolderManagement.create_folders(vae_base_path=vae_base_result_path, ae_base_path=ae_base_result_path)

    with mlflow.start_run(run_name=args.name, nested=True) as run:
        vae = VAE(args=args, base_result_path=vae_base_result_path)
        ae = AutoEncoder(args=args, base_results_path=ae_base_result_path)
