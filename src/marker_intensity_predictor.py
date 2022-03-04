from args import ArgumentParser
from Shared.folder_management import FolderManagement
import mlflow
from VAE.main import VAE
from AE.main import AutoEncoder
from pathlib import Path
from Plotting.plots import Plotting
from Shared.experiment_handler import ExperimentHandler
import os

#kfold vs random holdout
#https://stats.stackexchange.com/questions/283512/how-many-times-should-i-repeat-hold-out-cross-validation

vae_base_result_path = Path("results", "VAE")
ae_base_result_path = Path("results", "AE")
comparison_base_results_path = Path("results")

if __name__ == "__main__":
    os.environ[
        "AZURE_STORAGE_CONNECTION_STRING"] = ""

    args = ArgumentParser.get_args()
    # set tracking url
    if args.tracking_url is not None:
        mlflow.set_tracking_uri(args.tracking_url)

    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                                experiment_description=args.description)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

    mlflow.set_experiment(experiment_id=associated_experiment_id)
    FolderManagement.create_folders(vae_base_path=vae_base_result_path, ae_base_path=ae_base_result_path)
    # Start initial experiment
    with mlflow.start_run(run_name=args.run, nested=True, experiment_id=associated_experiment_id) as run:
        mlflow.log_param("Included Morphological Data", args.morph)
        mlflow.log_param("File", args.file)
        mlflow.log_param("Mode", args.mode)

        if args.mode == "vae":
            print("VAE only mode.")
            vae = VAE(args=args, base_result_path=vae_base_result_path, experiment_id=associated_experiment_id)

        elif args.mode == "ae":
            ae = AutoEncoder(args=args, base_results_path=ae_base_result_path, experiment_id=associated_experiment_id)

        else:
            vae = VAE(args=args, base_result_path=vae_base_result_path, experiment_id=associated_experiment_id)
            ae = AutoEncoder(args=args, base_results_path=ae_base_result_path, experiment_id=associated_experiment_id)

            # Start experiment which compares AE and VAE
            with mlflow.start_run(run_name="Comparison", nested=True,
                                  experiment_id=associated_experiment_id) as comparison:
                print("Comparing vae with ae.")
                plotter = Plotting(comparison_base_results_path, args)
                plotter.plot_r2_scores_comparison(ae_r2_scores=ae.evaluation.r2_scores,
                                                  vae_r2_scores=vae.evaluation.r2_scores)
