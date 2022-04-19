import argparse
from pathlib import Path
import pandas as pd
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
import mlflow
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.preprocessing.preprocessing import Preprocessing
from library.plotting.plots import Plotting
from typing import Optional
from library.mlflow_helper.reporter import Reporter
from library.vae.vae_imputer import VAEImputation
from library.mlflow_helper.run_handler import RunHandler
import time

base_path = Path("data_imputation_random_mean")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run",
                        type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="Assigns the run to a particular experiment. "
                             "If the experiment does not exists it will create a new one.",
                        default="Default", type=str)
    parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--model", "-m", action="store", nargs="+",
                        help="Specify experiment and run name from where to load the model",
                        type=str, required=True)
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--steps", action="store", help="The iterations for imputation",
                        default=3, required=False)

    return parser.parse_args()


def get_associated_experiment_id(args) -> Optional[str]:
    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    return associated_experiment_id


def get_model_experiment_id(args) -> str:
    model_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=args.model[0],
                                                                            create_experiment=False)
    if model_experiment_id is None:
        raise ValueError(f"Could not find experiment {args.model[0]}")

    return model_experiment_id


def get_model_run_id(run_handler: RunHandler, experiment_id: str, parent_run_id: str, run_name: str) -> str:
    model_run_id: str = run_handler.get_run_id_by_name(experiment_id=experiment_id, run_name=run_name,
                                                       parent_run_id=parent_run_id)

    if model_run_id is None:
        raise ValueError(f"Could not find run with name {args.model[1]}")

    return model_run_id


if __name__ == "__main__":
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    args = get_args()

    if len(args.model) != 3:
        raise ValueError(
            "Please specify the experiment as the first parameter and the run name as the second one and the specific model as the third.")

    requested_experiment = args.model[0]
    requested_parent_run_name = args.model[1]
    requested_run_name = args.model[2]

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    # Load model experiment and run id
    model_experiment_id: str = get_model_experiment_id(args)

    parent_run_id: str = run_handler.get_run_id_by_name(experiment_id=model_experiment_id,
                                                        run_name=requested_parent_run_name)
    model_run_id: str = get_model_run_id(run_handler=run_handler, experiment_id=model_experiment_id,
                                         run_name=requested_run_name, parent_run_id=parent_run_id)

    FolderManagement.create_directory(base_path)

    try:
        with mlflow.start_run(experiment_id=get_associated_experiment_id(args=args),
                              run_name=f"{args.run} Percentage {args.percentage} Steps {args.steps}") as run:

            iter_steps: int = int(args.steps)
            # Report
            mlflow.log_param("Percentage of replaced values", args.percentage)
            mlflow.log_param("Model location", f"{args.model[0]} {args.model[1]}")
            mlflow.log_param("File", args.file)
            mlflow.log_param("Seed", args.seed)
            mlflow.log_param("Iteration Steps", iter_steps)
            mlflow.set_tag("Percentage", args.percentage)
            mlflow.set_tag("Steps", iter_steps)

            # load model
            model = mlflow.keras.load_model(f"./mlruns/{model_experiment_id}/{model_run_id}/artifacts/model")

            # Load data
            cells, features = DataLoader.load_single_cell_data(args.file)
            # Use whole file
            test_data = pd.DataFrame(columns=features, data=Preprocessing.normalize(cells))

            ground_truth_data = test_data.copy()

            for step in range(1, iter_steps + 1):
                # Reconstructed by using the vae
                reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()

                # Just using the replaced values
                replaced_r2_scores: pd.DataFrame = pd.DataFrame()

                # Imputed by using the VAE
                imputed_r2_scores: pd.DataFrame = pd.DataFrame()

                with mlflow.start_run(experiment_id=get_associated_experiment_id(args=args), nested=True,
                                      run_name=f"{args.run} Percentage {args.percentage} Step {step}") as step_run:

                    mlflow.set_tag("Percentage", args.percentage)
                    mlflow.set_tag("Step", step)

                    for feature_to_impute in features:
                        imputed_r2_score, reconstructed_r2_score, replaced_r2_score = VAEImputation.impute_data(
                            model=model, ground_truth_data=ground_truth_data, feature_to_impute=feature_to_impute,
                            percentage=args.percentage, features=features, iter_steps=step)

                        # Add score to datasets
                        reconstructed_r2_scores = reconstructed_r2_scores.append({
                            "Marker": feature_to_impute,
                            "Score": reconstructed_r2_score,
                        }, ignore_index=True)

                        imputed_r2_scores = imputed_r2_scores.append({
                            "Marker": feature_to_impute,
                            "Score": imputed_r2_score
                        }, ignore_index=True)

                        replaced_r2_scores = replaced_r2_scores.append({
                            "Marker": feature_to_impute,
                            "Score": replaced_r2_score
                        }, ignore_index=True)

                    # Report results
                    plotter: Plotting = Plotting(base_path=base_path, args=args)
                    plotter.plot_scores(scores={"Ground Truth vs. Reconstructed": reconstructed_r2_scores,
                                                "Ground Truth vs. Imputed": imputed_r2_scores,
                                                "Ground Truth vs. Replaced": replaced_r2_scores},
                                        file_name=f"R2 Score Comparison Steps {iter_steps} Percentage {args.percentage}")

                    plotter.plot_scores(scores={"Imputed": imputed_r2_scores},
                                        file_name=f"R2 Imputed Scores Step {step} Percentage {args.percentage}")

                    Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=base_path,
                                              mlflow_folder="",
                                              prefix="reconstructed")
                    Reporter.report_r2_scores(r2_scores=replaced_r2_scores, save_path=base_path, mlflow_folder="",
                                              prefix="replaced")
                    Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=base_path, mlflow_folder="",
                                              prefix="imputed")

                    Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=base_path,
                                        file_name="Features")



    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
