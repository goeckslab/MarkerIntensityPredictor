import argparse
from library.data.data_loader import DataLoader
import pandas as pd
from library.data.folder_management import FolderManagement
from pathlib import Path
from library.mlflow_helper.experiment_handler import ExperimentHandler
import mlflow
from library.mlflow_helper.run_handler import RunHandler
from library.plotting.plots import Plotting
from library.evalation.evaluation import Evaluation
from library.mlflow_helper.reporter import Reporter

base_results_folder = Path("intra_run_comparison")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The summary run id to use for downl",
                        type=str)
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="Assigns the run to a particular experiment. "
                             "If the experiment does not exists it will create a new one.",
                        default="Default", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    FolderManagement.create_directory(path=base_results_folder)
    try:
        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
        run_handler = RunHandler = RunHandler(client=client)

        # The id of the associated
        associated_experiment_id = None

        experiment_name = args.experiment
        if experiment_name is not None:
            associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                                    create_experiment=False)

        # Experiment not found
        if associated_experiment_id is None:
            raise ValueError(
                f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

        mlflow.set_experiment(experiment_id=associated_experiment_id)

        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True, run_id=args.run) as run:
            with mlflow.start_run(experiment_id=associated_experiment_id, nested=True, run_name="Plots") as plot_run:
                experiment_handler.download_artifacts(base_save_path=base_results_folder, run=run)

                ae_r2_scores: pd.DataFrame = DataLoader.load_file(load_path=Path(base_results_folder, run.info.run_id),
                                                                  file_name="ae_r2_score.csv")
                en_r2_scores: pd.DataFrame = DataLoader.load_file(load_path=Path(base_results_folder, run.info.run_id),
                                                                  file_name="en_r2_score.csv")
                vae_r2_scores: pd.DataFrame = DataLoader.load_file(load_path=Path(base_results_folder, run.info.run_id),
                                                                   file_name="vae_r2_score.csv")
                me_vae_r2_scores: pd.DataFrame = DataLoader.load_file(
                    load_path=Path(base_results_folder, run.info.run_id),
                    file_name="me_vae_r2_score.csv")

                r2_scores = {"EN": en_r2_scores, "AE": ae_r2_scores, "VAE": vae_r2_scores,
                             "ME VAE": me_vae_r2_scores}

                features: list = DataLoader.load_file(load_path=Path(base_results_folder),
                                                      file_name="Features.csv")["Features"].to_list()

                relative_en_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_scores,
                                                                                              features=features,
                                                                                              reference_model="EN")
                relative_ae_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_scores,
                                                                                              features=features,
                                                                                              reference_model="AE")

                relative_me_vae_performance_scores = Evaluation.create_relative_score_performance(r2_scores=r2_scores,
                                                                                                  features=features,
                                                                                                  reference_model="ME VAE")

                plotter: Plotting = Plotting(args=args, base_path=base_results_folder)
                plotter.r2_score_differences(r2_scores=r2_scores, file_name="Score Differences",
                                             mlflow_directory="Plots")
                plotter.r2_scores_relative_performance(
                    relative_score_performance=relative_en_performance_scores,
                    features=features,
                    file_name="Relative EN Performance Difference", mlflow_directory="Plots")

                plotter.r2_scores_relative_performance(
                    relative_score_performance=relative_ae_performance_scores,
                    features=features,
                    file_name="Relative AE Performance Difference", mlflow_directory="Plots")

                plotter.r2_scores_relative_performance(
                    relative_score_performance=relative_me_vae_performance_scores,
                    features=features,
                    file_name="Relative ME VAE Performance Difference", mlflow_directory="Plots")

                absolute_performance_scores: pd.DataFrame = Evaluation.create_absolute_score_performance(
                    r2_scores=r2_scores, features=features)

                plotter.r2_scores_absolute_performance(absolute_score_performance=absolute_performance_scores,
                                                       file_name="Absolute Performance Comparison",
                                                       mlflow_directory="Plots")

                Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]),
                                    save_path=base_results_folder,
                                    file_name="Features")


    except BaseException as ex:
        raise
    finally:
        FolderManagement.delete_directory(path=base_results_folder)
