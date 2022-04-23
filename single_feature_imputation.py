import mlflow
from pathlib import Path
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
from library.preprocessing.replacements import Replacer
from library.preprocessing.preprocessing import Preprocessing
import pandas as pd
from library.knn.knn_imputation import KNNImputation
from library.plotting.plots import Plotting
from library.mlflow_helper.reporter import Reporter
from sklearn.metrics import r2_score
import time
from library.preprocessing.split import SplitHandler
from library.mlflow_helper.run_handler import RunHandler
from library.me_vae.me_vae_imputer import MEVAEImputation
from library.simple_imputer.simple_imputer import SimpleImputation
from library.vae.vae_imputer import VAEImputation
import argparse


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be evaluated",
                        default="Default", type=str)
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--file", action="store", required=True, help="The file to use for imputation")
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--model", "-m", action="store", nargs="+",
                        help="Specify experiment and run name from where to load the model",
                        type=str, required=True)
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--steps", action="store", help="The iterations for imputation",
                        default=3, required=False)

    return parser.parse_args()


def get_experiment_id(requested_experiment_name: str, experiment_handler: ExperimentHandler,
                      create_experiment: bool) -> str:
    model_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=requested_experiment_name,
                                                                            create_experiment=create_experiment)
    if model_experiment_id is None:
        raise ValueError(f"Could not find experiment {requested_experiment_name}")

    return model_experiment_id


def get_model_run_id(run_handler: RunHandler, experiment_id: str, parent_run_id: str, run_name: str) -> str:
    model_run_id: str = run_handler.get_run_id_by_name(experiment_id=experiment_id, run_name=run_name,
                                                       parent_run_id=parent_run_id)

    if model_run_id is None:
        raise ValueError(f"Could not find run with name {run_name}")

    return model_run_id


def start_simple_imputation(args, base_path: str):
    simple_imputer_results_path = Path(f"{base_path}", str(int(time.time_ns() / 1000)))
    run_name: str = "SI Imputation"
    if args.percentage >= 1:
        print("Percentage is greater than 100%. Returning")
        return

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found!")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(simple_imputer_results_path)

    try:
        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=f"{run_name} Percentage {args.percentage}") as run:
            mlflow.log_param("Percentage of replaced values", args.percentage)
            mlflow.log_param("Files", args.file)
            mlflow.log_param("Seed", args.seed)
            mlflow.set_tag("Percentage", args.percentage)

            cells, features = DataLoader.load_single_cell_data(args.file)

            train_data, test_data = SplitHandler.create_splits(cells=cells, features=features, create_val=False)

            train_data = pd.DataFrame(data=Preprocessing.normalize(data=train_data), columns=features)
            test_data = pd.DataFrame(data=Preprocessing.normalize(data=test_data), columns=features)

            r2_scores: pd.DataFrame = pd.DataFrame()
            for feature in features:
                print(f"Imputation started for feature {feature} ...")
                # Replace data for specific marker
                replaced_data, indexes = Replacer.replace_values_by_feature(data=test_data.copy(),
                                                                            feature_to_replace=feature,
                                                                            percentage=args.percentage,
                                                                            value_to_replace_with=0)

                # Select ground truth values for marker and indexes
                selected_ground_truth_values = test_data[feature].iloc[indexes].copy()

                # Impute missing values
                imputed_values = SimpleImputation.impute(train_data=train_data, test_data=replaced_data)

                # Select imputed values for marker and indexes
                selected_imputed_values = imputed_values[feature].iloc[indexes].copy()

                r2_scores = r2_scores.append(
                    {
                        "Marker": feature,
                        "Score": r2_score(selected_ground_truth_values, selected_imputed_values)
                    }, ignore_index=True
                )

            plotter = Plotting(base_path=simple_imputer_results_path, args=args)
            plotter.plot_scores(scores={"Simple Imputation": r2_scores}, file_name="Simple R2 Scores")
            plotter.plot_correlation(data_set=cells, file_name="Correlation")
            Reporter.report_r2_scores(r2_scores=r2_scores, save_path=simple_imputer_results_path, prefix="imputed")
            Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]),
                                save_path=simple_imputer_results_path,
                                file_name="Features")



    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(simple_imputer_results_path)


def start_knn_imputation(base_path: str, args):
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "KNN Imputation"

    if args.percentage >= 1:
        print("Percentage is greater than 100%. Returning")
        return

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # The id of the associated
    associated_experiment_id = None

    experiment_name = args.experiment
    if experiment_name is not None:
        associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    # Experiment not found
    if associated_experiment_id is None:
        raise ValueError(
            f"Experiment {experiment_name} not found!")

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(base_path)

    try:
        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=f"{run_name} Percentage {args.percentage}") as run:
            mlflow.log_param("Percentage of replaced values", args.percentage)
            mlflow.log_param("Files", args.file)
            mlflow.log_param("Seed", args.seed)
            mlflow.set_tag("Percentage", args.percentage)

            cells, features = DataLoader.load_single_cell_data(args.file)

            train_data, test_data = SplitHandler.create_splits(cells=cells, features=features, create_val=False)

            train_data = pd.DataFrame(data=Preprocessing.normalize(data=train_data), columns=features)
            test_data = pd.DataFrame(data=Preprocessing.normalize(data=test_data), columns=features)

            r2_scores: pd.DataFrame = pd.DataFrame()
            for feature in features:
                print(f"Imputation started for feature {feature} ...")
                # Replace data for specific marker
                replaced_data, indexes = Replacer.replace_values_by_feature(data=test_data.copy(),
                                                                            feature_to_replace=feature,
                                                                            percentage=args.percentage,
                                                                            value_to_replace_with=0)

                # Select ground truth values for marker and indexes
                selected_ground_truth_values = test_data[feature].iloc[indexes].copy()

                # Impute missing values
                imputed_values = KNNImputation.impute(train_data=train_data, test_data=replaced_data)

                # Select imputed values for marker and indexes
                selected_imputed_values = imputed_values[feature].iloc[indexes].copy()

                r2_scores = r2_scores.append(
                    {
                        "Marker": feature,
                        "Score": r2_score(selected_ground_truth_values, selected_imputed_values)
                    }, ignore_index=True
                )

            plotter = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores={"KNN": r2_scores}, file_name="KNN R2 Scores")
            plotter.plot_correlation(data_set=cells, file_name="Correlation")
            Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path, prefix="imputed")
            Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=base_path,
                                file_name="Features")




    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)


def start_me_imputation(args, base_path: str):
    me_vae_results_path: Path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "ME VAE Imputation"
    # Where to store the results
    save_experiment_name: str = args.experiment

    # Where to find the model. This can be a different experiment
    requested_model_experiment_name: str = args.model[0]
    requested_model_parent_run_name: str = args.model[1]
    requested_model_run_name: str = "ME VAE"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    # Load model experiment and run id
    model_experiment_id: str = get_experiment_id(requested_experiment_name=requested_model_experiment_name,
                                                 experiment_handler=experiment_handler, create_experiment=False)
    parent_run_id: str = run_handler.get_run_id_by_name(experiment_id=model_experiment_id,
                                                        run_name=requested_model_parent_run_name)
    model_run_id: str = get_model_run_id(run_handler=run_handler, experiment_id=model_experiment_id,
                                         run_name=requested_model_run_name, parent_run_id=parent_run_id)

    FolderManagement.create_directory(me_vae_results_path)

    save_experiment_id: str = get_experiment_id(requested_experiment_name=save_experiment_name,
                                                experiment_handler=experiment_handler,
                                                create_experiment=False)

    try:

        iter_steps: int = int(args.steps)

        # load model
        model = mlflow.keras.load_model(f"./mlruns/{model_experiment_id}/{model_run_id}/artifacts/model")

        # Load data
        cells, features = DataLoader.load_single_cell_data(args.file)

        # Split and normalize ground truth values for reference
        ground_truth_data = cells.copy()
        ground_truth_marker_data, ground_truth_morph_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            ground_truth_data)
        ground_truth_morph_data = pd.DataFrame(columns=ground_truth_morph_data.columns,
                                               data=Preprocessing.normalize(ground_truth_morph_data))
        ground_truth_marker_data = pd.DataFrame(columns=ground_truth_marker_data.columns,
                                                data=Preprocessing.normalize(ground_truth_marker_data))

        # Split and normalize data
        marker_data, morph_data = SplitHandler.split_dataset_into_markers_and_morph_features(cells)
        morph_data = pd.DataFrame(columns=morph_data.columns, data=Preprocessing.normalize(morph_data))
        marker_data = pd.DataFrame(columns=marker_data.columns, data=Preprocessing.normalize(marker_data))

        for step in range(1, iter_steps + 1):

            # Reconstructed by using the vae
            reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()

            # Just using the replaced values
            replaced_r2_scores: pd.DataFrame = pd.DataFrame()

            # Imputed by using the VAE
            imputed_r2_scores: pd.DataFrame = pd.DataFrame()

            with mlflow.start_run(experiment_id=save_experiment_id, nested=True,
                                  run_name=f"{run_name} Percentage {args.percentage} Step {step}") as step_run:
                # Report
                mlflow.log_param("Percentage of replaced values", args.percentage)
                mlflow.log_param("Model location", f"{args.model[0]} {args.model[1]}")
                mlflow.log_param("File", args.file)
                mlflow.log_param("Seed", args.seed)
                mlflow.log_param("Iteration Steps", iter_steps)
                mlflow.set_tag("Percentage", args.percentage)
                mlflow.set_tag("Steps", iter_steps)
                mlflow.set_tag("Percentage", args.percentage)
                mlflow.set_tag("Step", step)

                for marker_to_impute in marker_data.columns:
                    imputed_r2_score, reconstructed_r2_score, replaced_r2_score = MEVAEImputation.impute_data_by_feature(
                        model=model,
                        feature_to_impute=marker_to_impute,
                        marker_data=marker_data.copy(),
                        morph_data=morph_data.copy(),
                        ground_truth_marker_data=ground_truth_marker_data.copy(),
                        ground_truth_morph_data=ground_truth_morph_data.copy(),
                        percentage=args.percentage,
                        iter_steps=step, features=features)

                    # Add score to datasets
                    reconstructed_r2_scores = reconstructed_r2_scores.append({
                        "Marker": marker_to_impute,
                        "Score": reconstructed_r2_score,
                    }, ignore_index=True)

                    imputed_r2_scores = imputed_r2_scores.append({
                        "Marker": marker_to_impute,
                        "Score": imputed_r2_score
                    }, ignore_index=True)

                    replaced_r2_scores = replaced_r2_scores.append({
                        "Marker": marker_to_impute,
                        "Score": replaced_r2_score
                    }, ignore_index=True)

                for morph_to_impute in morph_data.columns:
                    imputed_r2_score, reconstructed_r2_score, replaced_r2_score = MEVAEImputation.impute_data_by_feature(
                        model=model,
                        feature_to_impute=morph_to_impute,
                        marker_data=marker_data.copy(),
                        morph_data=morph_data.copy(),
                        ground_truth_marker_data=ground_truth_marker_data.copy(),
                        ground_truth_morph_data=ground_truth_morph_data.copy(),
                        percentage=args.percentage,
                        iter_steps=step, features=features)

                    # Add score to datasets
                    reconstructed_r2_scores = reconstructed_r2_scores.append({
                        "Marker": morph_to_impute,
                        "Score": reconstructed_r2_score,
                    }, ignore_index=True)

                    imputed_r2_scores = imputed_r2_scores.append({
                        "Marker": morph_to_impute,
                        "Score": imputed_r2_score
                    }, ignore_index=True)

                    replaced_r2_scores = replaced_r2_scores.append({
                        "Marker": morph_to_impute,
                        "Score": replaced_r2_score
                    }, ignore_index=True)

                # Report and plot results
                plotter: Plotting = Plotting(base_path=me_vae_results_path, args=args)
                plotter.plot_scores(scores={"Ground Truth vs. Reconstructed": reconstructed_r2_scores,
                                            "Ground Truth vs. Imputed": imputed_r2_scores,
                                            "Ground Truth vs. Replaced": replaced_r2_scores},
                                    file_name=f"R2 Score Comparison Step {step} Percentage {args.percentage}")

                plotter.plot_scores(scores={"Imputed": imputed_r2_scores},
                                    file_name=f"R2 Imputed Scores Step {step} Percentage {args.percentage}")

                Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=me_vae_results_path,
                                          mlflow_folder="",
                                          prefix="ground_truth")
                Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=me_vae_results_path,
                                          mlflow_folder="",
                                          prefix="imputed")
                Reporter.report_r2_scores(r2_scores=replaced_r2_scores, save_path=me_vae_results_path,
                                          mlflow_folder="",
                                          prefix="replaced")
                Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]),
                                    save_path=me_vae_results_path,
                                    file_name="Features")





    except:
        raise
    finally:
        FolderManagement.delete_directory(me_vae_results_path)


def start_se_imputation(base_path: str, args):
    se_imputation_base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "VAE Imputation"
    if len(args.model) != 2:
        raise ValueError(
            "Please specify the experiment as the first parameter and the run name as the second one and the specific model as the third.")

    save_experiment_name: str = args.experiment

    requested_model_experiment_name: str = args.model[0]
    requested_model_parent_run_name: str = args.model[1]
    requested_model_run_name: str = "VAE"

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
    run_handler: RunHandler = RunHandler(client=client)

    # Load model experiment and run id
    model_experiment_id: str = get_experiment_id(requested_experiment_name=requested_model_experiment_name,
                                                 experiment_handler=experiment_handler, create_experiment=False)

    parent_run_id: str = run_handler.get_run_id_by_name(experiment_id=model_experiment_id,
                                                        run_name=requested_model_parent_run_name)
    model_run_id: str = get_model_run_id(run_handler=run_handler, experiment_id=model_experiment_id,
                                         run_name=requested_model_run_name, parent_run_id=parent_run_id)

    FolderManagement.create_directory(se_imputation_base_path)

    save_experiment_id: str = get_experiment_id(requested_experiment_name=save_experiment_name,
                                                experiment_handler=experiment_handler, create_experiment=True)

    try:

        iter_steps: int = int(args.steps)

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

            with mlflow.start_run(experiment_id=save_experiment_id, nested=True,
                                  run_name=f"{run_name} Percentage {args.percentage} Step {step}") as step_run:
                # Report
                mlflow.log_param("Percentage of replaced values", args.percentage)
                mlflow.log_param("Model location", f"{args.model[0]} {args.model[1]}")
                mlflow.log_param("File", args.file)
                mlflow.log_param("Seed", args.seed)
                mlflow.log_param("Iteration Steps", iter_steps)
                mlflow.set_tag("Percentage", args.percentage)
                mlflow.set_tag("Steps", iter_steps)
                mlflow.set_tag("Percentage", args.percentage)
                mlflow.set_tag("Step", step)

                for feature_to_impute in features:
                    imputed_r2_score, reconstructed_r2_score, replaced_r2_score = VAEImputation.impute_data_by_feature(
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
                plotter: Plotting = Plotting(base_path=se_imputation_base_path, args=args)
                plotter.plot_scores(scores={"Ground Truth vs. Reconstructed": reconstructed_r2_scores,
                                            "Ground Truth vs. Imputed": imputed_r2_scores,
                                            "Ground Truth vs. Replaced": replaced_r2_scores},
                                    file_name=f"R2 Score Comparison Steps {iter_steps} Percentage {args.percentage}")

                plotter.plot_scores(scores={"Imputed": imputed_r2_scores},
                                    file_name=f"R2 Imputed Scores Step {step} Percentage {args.percentage}")

                Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=se_imputation_base_path,
                                          mlflow_folder="",
                                          prefix="reconstructed")
                Reporter.report_r2_scores(r2_scores=replaced_r2_scores, save_path=se_imputation_base_path,
                                          mlflow_folder="",
                                          prefix="replaced")
                Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=se_imputation_base_path,
                                          mlflow_folder="",
                                          prefix="imputed")

                Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]),
                                    save_path=se_imputation_base_path,
                                    file_name="Features")



    except:
        raise
    finally:
        FolderManagement.delete_directory(se_imputation_base_path)


if __name__ == "__main__":
    args = get_args()

    base_results_path = Path("imputation")
    si_path = Path(base_results_path, "si")
    knn_path = Path(base_results_path, "knn")
    se_path = Path(base_results_path, "se")
    me_path = Path(base_results_path, "me")

    FolderManagement.create_directory(path=base_results_path)

    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        experiment_handler: ExperimentHandler = ExperimentHandler(client=client)
        save_experiment_id: str = get_experiment_id(requested_experiment_name=args.experiment, create_experiment=True,
                                                    experiment_handler=experiment_handler)

        percentage = args.percentage * 100
        with mlflow.start_run(experiment_id=save_experiment_id,
                              run_name=f"{args.run} Steps: {args.steps} {percentage} %", nested=True) as run:
            start_simple_imputation(args=args, base_path=str(si_path))
            start_knn_imputation(args=args, base_path=str(knn_path))
            start_se_imputation(args=args, base_path=str(se_path))
            start_me_imputation(args=args, base_path=str(me_path))


    except BaseException as ex:
        raise
    finally:
        FolderManagement.delete_directory(base_results_path)
