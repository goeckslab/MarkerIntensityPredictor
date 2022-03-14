import mlflow
import logging
from pathlib import Path
import pandas as pd
from library.data.folder_management import FolderManagement
import argparse
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.mlflow_helper.reporter import Reporter
from library.plotting.plots import Plotting
from library.data.data_loader import DataLoader

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=True,
                        help="The name of the experiment which should be evaluated", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")

    return parser.parse_args()


class ExperimentComparer:
    # All runs to compare
    runs: list = []
    client = mlflow.tracking.MlflowClient()
    base_path = Path("run_comparison")
    # The user given experiment name
    experiment_name: str
    # The experiment id
    experiment_id: str
    experiment_handler: ExperimentHandler

    download_directory: Path
    ae_directory: Path
    vae_directory: Path

    def __init__(self, experiment_name: str):
        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        self.experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

        self.experiment_name = experiment_name
        self.experiment_id = self.experiment_handler.get_experiment_id_by_name(experiment_name=self.experiment_name,
                                                                               create_experiment=False)

        if self.experiment_id is None:
            raise ValueError(f"Could not find experiment with name {experiment_name}")

        self.download_directory = FolderManagement.create_directory(Path(self.base_path, "runs"))
        self.ae_directory = FolderManagement.create_directory(Path(self.download_directory, "AE"))
        self.vae_directory = FolderManagement.create_directory(Path(self.download_directory, "VAE"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if self.base_path is not None:
            FolderManagement.delete_directory(self.base_path)

    def start_comparison(self):
        if self.experiment_id is None:
            print(f"Could not find an experiment with the given name: {self.experiment_name}")
            return

        with mlflow.start_run(run_name="Run Comparison", experiment_id=self.experiment_id) as run:
            # Collect all experiments based on the search tag
            self.runs = self.experiment_handler.get_vae_ae_runs(experiment_id=self.experiment_id)

            if len(self.runs) == 0:
                print(f"No runs found.")
                print("Resources are being cleaned up.")
                # self.client.delete_run(run.info.run_id)
                return

            print(f"Found {len(self.runs)} runs.")

            # Select all auto encoder runs
            ae_runs = [run for run in self.runs if
                       run.data.tags.get("Model") == "AE" and run.data.tags.get('mlflow.parentRunId') is not None]

            # Select all vae runs
            vae_runs = [run for run in self.runs if
                        run.data.tags.get("Model") == "VAE" and run.data.tags.get('mlflow.parentRunId') is not None]

            # Download all artifacts
            self.__download_artifacts(ae_runs=ae_runs, vae_runs=vae_runs)

            ae_mean_scores, ae_combined_scores = DataLoader.load_r2_scores_for_model(self.ae_directory)
            vae_mean_scores, vae_combined_scores = DataLoader.load_r2_scores_for_model(self.vae_directory)
            mlflow.log_param("AE_marker_count", ae_mean_scores.shape[0])
            mlflow.log_param("VAE_marker_count", vae_mean_scores.shape[0])
            Reporter.report_r2_score_mean_difference(ae_combined_scores.diff().mean(), self.ae_directory)
            Reporter.report_r2_score_mean_difference(vae_combined_scores.diff().mean(), self.vae_directory)

            self.__report_r2_scores({
                "ae_mean": ae_mean_scores,
                "ae_combined": ae_combined_scores,
                "vae_mean": vae_mean_scores,
                "vae_combined": vae_combined_scores
            })

            encoding_layer_weights: pd.DataFrame = DataLoader.load_layer_weights(self.ae_directory,
                                                                                 "layer_encoding_h1_weights.csv")
            decoding_layer_weights: pd.DataFrame = DataLoader.load_layer_weights(self.ae_directory,
                                                                                 "layer_decoding_output_weights.csv")

            plotter = Plotting(self.base_path, args=args)

            plotter.compare_vae_to_ae_scores(ae_scores=ae_mean_scores, vae_scores=vae_mean_scores)
            plotter.r2_score_distribution(combined_r2_scores=ae_combined_scores,
                                          comparing_r2_scores=vae_combined_scores,
                                          title="AE", comparing_title="VAE", file_name="r2_distribution")
            plotter.plot_weights_distribution(encoding_layer_weights, "encoding")
            plotter.plot_weights_distribution(decoding_layer_weights, "decoding")

            plotter.plot_r2_score_differences(pd.DataFrame(columns=["Marker", "Score"],
                                                           data={'Marker': ae_combined_scores.columns,
                                                                 'Score': ae_combined_scores.diff().mean().values}),
                                              "ae")
            plotter.plot_r2_score_differences(pd.DataFrame(columns=["Marker", "Score"],
                                                           data={'Marker': ae_combined_scores.columns,
                                                                 'Score': vae_combined_scores.diff().mean().values}),
                                              "vae")
            # plotter.plot_weights(weights=encoding_layer_weights.mean().T,
            #                     markers=list(encoding_layer_weights.columns.values),
            #                     mlflow_directory="", fig_name="layer_encoding_h1_weights")
            # plotter.plot_weights(decoding_layer_weights.mean().T, markers=list(encoding_layer_weights.columns.values),
            #                     mlflow_directory="", fig_name="layer_decoding_weights")

            self.__log_information()

    def __log_information(self):
        mlflow.log_param("Included Runs", len(self.runs))
        mlflow.log_param("Used Run Ids",
                         [x.info.run_id for x in self.runs])

    def __download_artifacts(self, ae_runs: [], vae_runs: []):
        # Download ae evaluation files
        self.experiment_handler.download_artifacts(save_path=self.ae_directory, runs=ae_runs,
                                                   mlflow_folder="Evaluation")

        self.experiment_handler.download_artifacts(save_path=self.vae_directory, runs=vae_runs,
                                                   mlflow_folder="Evaluation")

        self.experiment_handler.download_artifacts(save_path=self.ae_directory, runs=ae_runs,
                                                   mlflow_folder="AE")

        self.experiment_handler.download_artifacts(save_path=self.vae_directory, runs=vae_runs,
                                                   mlflow_folder="VAE")

    def __report_r2_scores(self, scores: {}):
        for key, scores in scores.items():
            Reporter.report_r2_scores(scores, save_path=Path(self.base_path, "runs"),
                                      mlflow_folder="", prefix=key)


if __name__ == "__main__":
    args = get_args()

    comparer = ExperimentComparer(args.experiment)
    comparer.start_comparison()
else:
    raise "Tool is meant to be executed as standalone"
