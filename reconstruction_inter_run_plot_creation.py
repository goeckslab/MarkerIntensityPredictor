import mlflow
import logging
from pathlib import Path
import pandas as pd
from library.data.folder_management import FolderManagement
import argparse
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.mlflow_helper.run_handler import RunHandler
from library.mlflow_helper.reporter import Reporter
from library.plotting.plots import Plotting
from library.data.data_loader import DataLoader
from library.evalation.evaluation import Evaluation

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
    base_path = Path("inter_run_comparison")
    # The user given experiment name
    experiment_name: str
    # The experiment id
    experiment_id: str
    experiment_handler: ExperimentHandler
    run_handler: RunHandler

    download_directory: Path

    def __init__(self, experiment_name: str):

        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        self.experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
        self.run_handler: RunHandler = RunHandler(client=client)

        self.experiment_name = experiment_name
        self.experiment_id = self.experiment_handler.get_experiment_id_by_name(experiment_name=self.experiment_name,
                                                                               create_experiment=False)

        if self.experiment_id is None:
            raise ValueError(f"Could not find experiment with name {experiment_name}")

        self.download_directory = FolderManagement.create_directory(Path(self.base_path, "runs"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if self.base_path is not None:
            FolderManagement.delete_directory(self.base_path)

    def start_comparison(self):
        if self.experiment_id is None:
            print(f"Could not find an experiment with the given name: {self.experiment_name}")
            return

        with mlflow.start_run(run_name="Inter Run Summary", experiment_id=self.experiment_id) as run:
            # Collect all experiments based on the search tag
            self.runs = self.run_handler.get_summary_runs(experiment_id=self.experiment_id)

            if len(self.runs) == 0:
                print(f"No runs found.")
                print("Resources are being cleaned up.")
                return

            mlflow.log_param("Included Runs", len(self.runs))
            mlflow.log_param("Used Run Ids",
                             [x.info.run_id for x in self.runs])

            print(f"Found {len(self.runs)} runs.")

            self.run_handler.download_artifacts(base_save_path=self.download_directory, runs=self.runs)

            ae_mean_scores, ae_combined_scores = DataLoader.load_r2_scores_for_model(self.download_directory,
                                                                                     "ae_r2_score.csv")
            vae_mean_scores, vae_combined_scores = DataLoader.load_r2_scores_for_model(self.download_directory,
                                                                                       "vae_r2_score.csv")
            en_mean_scores, en_combined_scores = DataLoader.load_r2_scores_for_model(self.download_directory,
                                                                                     "en_r2_score.csv")
            me_vae_mean_scores, me_vae_combined_scores = DataLoader.load_r2_scores_for_model(self.download_directory,
                                                                                             "me_vae_r2_score.csv")

            mlflow.log_param("AE feature count", ae_mean_scores.shape[0])
            mlflow.log_param("VAE feature count", vae_mean_scores.shape[0])
            mlflow.log_param("EN feature count", en_mean_scores.shape[0])
            mlflow.log_param("ME VAE feature count", me_vae_mean_scores.shape[0])

            # Report mean differences
            ae_mean_difference_scores = pd.DataFrame(columns=["Markers", "Score"])
            ae_mean_difference_scores["Markers"] = ae_combined_scores.columns
            ae_mean_difference_scores["Score"] = ae_combined_scores.diff().mean().values

            Reporter.report_r2_score_mean_difference(r2score_difference=ae_mean_difference_scores,
                                                     save_path=self.base_path, prefix="ae")

            vae_mean_difference_scores = pd.DataFrame(columns=["Markers", "Score"])
            vae_mean_difference_scores["Markers"] = vae_combined_scores.columns
            vae_mean_difference_scores["Score"] = vae_combined_scores.diff().mean().values
            Reporter.report_r2_score_mean_difference(r2score_difference=vae_mean_difference_scores,
                                                     save_path=self.base_path, prefix="vae")

            en_mean_difference_scores = pd.DataFrame(columns=["Markers", "Score"])
            en_mean_difference_scores["Markers"] = en_combined_scores.columns
            en_mean_difference_scores["Score"] = en_combined_scores.diff().mean().values
            Reporter.report_r2_score_mean_difference(r2score_difference=en_mean_difference_scores,
                                                     save_path=self.base_path, prefix="en")

            me_vae_mean_difference_scores = pd.DataFrame(columns=["Markers", "Score"])
            me_vae_mean_difference_scores["Markers"] = me_vae_combined_scores.columns
            me_vae_mean_difference_scores["Score"] = me_vae_combined_scores.diff().mean().values
            Reporter.report_r2_score_mean_difference(r2score_difference=me_vae_mean_difference_scores,
                                                     save_path=self.base_path, prefix="me_vae")

            self.__report_r2_scores({
                "ae_mean": ae_mean_scores,
                "ae_combined": ae_combined_scores,
                "vae_mean": vae_mean_scores,
                "vae_combined": vae_combined_scores,
                "en_mean": en_mean_scores,
                "en_combined": en_combined_scores,
                "me_vae_mean": me_vae_mean_scores,
                "me_vae_combined": me_vae_combined_scores
            })

            plotter = Plotting(self.base_path, args=args)

            r2_mean_scores = {"EN": en_mean_scores, "AE": ae_mean_scores, "VAE": vae_mean_scores,
                              "ME VAE": me_vae_mean_scores}

            plotter.r2_score_differences(r2_scores=r2_mean_scores, file_name="R2 Mean Difference",
                                         mlflow_directory="Plots")

            # Plot mean scores
            plotter.plot_scores(scores=r2_mean_scores, file_name="R2 Mean Scores", mlflow_directory="Plots")

            # Plot distribution
            plotter.r2_scores_distribution(
                {"EN": en_combined_scores, "AE": ae_combined_scores, "VAE": vae_combined_scores,
                 "ME VAE": me_vae_combined_scores},
                file_name=f"{args.experiment} R2 Score Distribution", mlflow_directory="Plots")

            plotter.plot_scores(scores={"AE": pd.DataFrame(columns=["Marker", "Score"],
                                                           data={
                                                               'Marker': ae_combined_scores.columns,
                                                               'Score': ae_combined_scores.diff().mean().values})},
                                file_name=f"{args.experiment} AE Mean Difference", mlflow_directory="Plots")

            plotter.plot_scores(scores={"VAE": pd.DataFrame(columns=["Marker", "Score"],
                                                            data={
                                                                'Marker': vae_combined_scores.columns,
                                                                'Score': vae_combined_scores.diff().mean().values})},
                                file_name=f"{args.experiment} VAE Mean Difference", mlflow_directory="Plots")

            plotter.plot_scores(scores={"EN": pd.DataFrame(columns=["Marker", "Score"],
                                                           data={
                                                               'Marker': en_combined_scores.columns,
                                                               'Score': en_combined_scores.diff().mean().values})},
                                file_name=f"{args.experiment} EN Mean Difference", mlflow_directory="Plots")

            plotter.plot_scores(scores={"ME VAE": pd.DataFrame(columns=["Marker", "Score"],
                                                               data={
                                                                   'Marker': me_vae_combined_scores.columns,
                                                                   'Score': me_vae_combined_scores.diff().mean().values})},
                                file_name=f"{args.experiment} ME VAE Mean Difference", mlflow_directory="Plots")

            features: list = DataLoader.load_file(load_path=Path(self.download_directory, self.runs[0].info.run_id),
                                                  file_name="Features.csv")["Features"].to_list()

            relative_en_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_mean_scores,
                                                                                          features=features,
                                                                                          reference_model="EN")
            relative_ae_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_mean_scores,
                                                                                          features=features,
                                                                                          reference_model="AE")

            relative_me_vae_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_mean_scores,
                                                                                              features=features,
                                                                                              reference_model="ME VAE")

            plotter.r2_scores_relative_performance(
                relative_score_performance=relative_en_performance_scores,
                features=features,
                file_name=f"{args.experiment} Relative EN Performance Difference",
                mlflow_directory="Plots"
            )

            plotter.r2_scores_relative_performance(
                relative_score_performance=relative_ae_performance_scores,
                features=features,
                file_name=f"{args.experiment} Relative AE Performance Difference",
                mlflow_directory="Plots"
            )

            plotter.r2_scores_relative_performance(
                relative_score_performance=relative_me_vae_performance_scores,
                features=features,
                file_name=f"{args.experiment} Relative ME VAE Performance Difference",
                mlflow_directory="Plots"
            )

            absolute_performance_scores: dict = {args.experiment: Evaluation.create_absolute_score_performance(
                r2_scores=r2_mean_scores, features=features)}

            plotter.r2_scores_combined_bar_plot(r2_scores=absolute_performance_scores,
                                                file_name=f"{args.experiment} Absolute Performance Comparison",
                                                mlflow_directory="Plots")

            Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=self.base_path,
                                file_name="Features")

    def __report_r2_scores(self, scores: {}):
        for key, scores in scores.items():
            Reporter.report_r2_scores(scores, save_path=Path(self.base_path, "runs"),
                                      mlflow_folder="", prefix=f"{key}")


if __name__ == "__main__":
    args = get_args()

    with ExperimentComparer(experiment_name=args.experiment) as comparer:
        comparer.start_comparison()
else:
    raise "Tool is meant to be executed as standalone"
