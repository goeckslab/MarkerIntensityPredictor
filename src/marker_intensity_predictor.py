from args import ArgumentParser
from Shared.folder_management import FolderManagement
from Shared.data_loader import DataLoader
import mlflow
from VAE.main import VAE
from AE.main import AutoEncoder
from pathlib import Path
from Plotting.plots import Plotting

vae_base_result_path = Path("results", "VAE")
ae_base_result_path = Path("results", "AE")
comparison_base_results_path = Path("results")

if __name__ == "__main__":
    args = ArgumentParser.get_args()

    FolderManagement.create_folders(vae_base_path=vae_base_result_path, ae_base_path=ae_base_result_path)
    # Start initial experiment
    with mlflow.start_run(run_name=args.experiment, nested=True) as run:
        mlflow.log_param("Included Morphological Data", args.morph)
        mlflow.log_param("File", args.file)
        mlflow.log_param("Mode", args.mode)
        mlflow.set_tag("Group", args.group)

        if args.mode == "none":
            vae = VAE(args=args, base_result_path=vae_base_result_path)
            ae = AutoEncoder(args=args, base_results_path=ae_base_result_path)

        elif args.mode == "vae":
            vae = VAE(args=args, base_result_path=vae_base_result_path)

        elif args.mode == "ae":
            ae = AutoEncoder(args=args, base_results_path=ae_base_result_path)

        if args.mode == "none":
            # Start experiment which compares AE and VAE
            with mlflow.start_run(run_name="Comparison", nested=True) as comparison:
                print("Comparing vae with ae.")
                plotter = Plotting(comparison_base_results_path)
                plotter.plot_r2_scores_comparison(ae_r2_scores=ae.evaluation.r2_scores,
                                                  vae_r2_scores=vae.evaluation.r2_scores)
