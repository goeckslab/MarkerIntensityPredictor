import argparse
from pathlib import Path
from library.preprocessing.split import SplitHandler
import pandas as pd
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
import mlflow
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.preprocessing.preprocessing import Preprocessing
from sklearn.metrics import r2_score
from library.plotting.plots import Plotting
from typing import Optional
from library.mlflow_helper.reporter import Reporter
from library.predictions.predictions import Predictions
from sklearn.metrics import accuracy_score, precision_score

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
                        default=0.2, required=False)
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


def get_model_run_id(args, model_experiment_id: str) -> str:
    model_run_id: str = experiment_handler.get_run_id_by_name(experiment_id=model_experiment_id, run_name=args.model[1])

    if model_run_id is None:
        raise ValueError(f"Could not find run with name {args.model[1]}")

    return model_run_id





if __name__ == "__main__":
    args = get_args()
    # Percentage of data replaced by random sampling
    fraction: float = 1 - float(args.percentage)

    if len(args.model) != 2:
        raise ValueError("Please specify the experiment as the first parameter and the run name as the second one!")

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    # Load model experiment and run id
    model_experiment_id: str = get_model_experiment_id(args)
    model_run_id: str = get_model_run_id(args=args, model_experiment_id=model_experiment_id)

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

            # load model
            model = mlflow.keras.load_model(f"./mlruns/{model_experiment_id}/{model_run_id}/artifacts/model")

            # Load data
            cells, markers = DataLoader.load_marker_data(args.file)

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

            # Reconstructed by using the vae
            reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()

            # Just using the replaced values
            replaced_r2_scores: pd.DataFrame = pd.DataFrame()

            # Imputed by using the VAE
            imputed_r2_scores: pd.DataFrame = pd.DataFrame()

            for marker_to_impute in marker_data.columns:
                # Make a fresh copy, to start with the ground truth data
                working_marker_data = marker_data.copy()
                working_morph_data = morph_data.copy()

                # Replace % of the data provided by the args
                working_marker_data[marker_to_impute] = working_marker_data[marker_to_impute].sample(frac=fraction,
                                                                                                     replace=False)

                indexes = working_marker_data[working_marker_data[marker_to_impute].isna()].index

                # values = np.random.normal(loc=marker_mean, scale=marker_std,
                #                          size=test_data[marker_to_impute].isna().sum())
                values = [0] * working_marker_data[marker_to_impute].isna().sum()
                working_marker_data[marker_to_impute].fillna(pd.Series(values, index=indexes), inplace=True)

                imputed_marker_data: pd.DataFrame = working_marker_data.iloc[indexes].copy()
                imputed_morph_data: pd.DataFrame = working_morph_data.iloc[indexes].copy()

                # Iterate to impute
                for i in range(iter_steps):
                    # Predict embeddings and mean
                    mean, log_var, z = model.encoder.predict([imputed_marker_data, imputed_morph_data])

                    # Create reconstructed date
                    reconstructed_data = pd.DataFrame(columns=markers, data=model.decoder.predict(mean))

                    # Overwrite imputed data with reconstructed data. Drop reconstructed morph data
                    imputed_marker_data = reconstructed_data.drop(
                        ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"],
                        axis=1)

                # Derive from ground truth dataset, to only compare imputation values.
                new_imputed_data = ground_truth_data.copy()
                new_imputed_data.loc[imputed_marker_data.index, :] = imputed_marker_data[:]

                # Reconstruct unmodified test data
                encoded_data, reconstructed_data = Predictions.encode_decode_me_vae_data(encoder=model.encoder,
                                                                                         decoder=model.decoder,
                                                                                         data=[ground_truth_marker_data,
                                                                                               ground_truth_morph_data],
                                                                                         markers=markers,
                                                                                         use_mlflow=False)

                reconstructed_r2_scores = reconstructed_r2_scores.append({
                    "Marker": marker_to_impute,
                    "Score": r2_score(ground_truth_marker_data[marker_to_impute].iloc[indexes],
                                      reconstructed_data[marker_to_impute].iloc[indexes]),
                }, ignore_index=True)

                imputed_r2_scores = imputed_r2_scores.append({
                    "Marker": marker_to_impute,
                    "Score": r2_score(ground_truth_marker_data[marker_to_impute].iloc[indexes],
                                      imputed_marker_data[marker_to_impute])
                }, ignore_index=True)

                replaced_r2_scores = replaced_r2_scores.append({
                    "Marker": marker_to_impute,
                    "Score": r2_score(ground_truth_marker_data[marker_to_impute].iloc[indexes],
                                      working_marker_data[marker_to_impute].iloc[indexes])
                }, ignore_index=True)

            for morphological_feature_to_impute in morph_data.columns:
                # Make a fresh copy, to start with the ground truth data
                working_marker_data = marker_data.copy()
                working_morph_data = morph_data.copy()

                # Replace % of the data provided by the args
                working_morph_data[morphological_feature_to_impute] = working_morph_data[
                    morphological_feature_to_impute].sample(frac=fraction,
                                                            replace=False)

                indexes = working_morph_data[working_morph_data[morphological_feature_to_impute].isna()].index

                # values = np.random.normal(loc=marker_mean, scale=marker_std,
                #                          size=test_data[marker_to_impute].isna().sum())
                values = [0] * working_morph_data[morphological_feature_to_impute].isna().sum()
                working_morph_data[morphological_feature_to_impute].fillna(pd.Series(values, index=indexes),
                                                                           inplace=True)

                imputed_marker_data: pd.DataFrame = working_marker_data.iloc[indexes].copy()
                imputed_morph_data: pd.DataFrame = working_morph_data.iloc[indexes].copy()

                # Iterate to impute
                for i in range(iter_steps):
                    # Predict embeddings and mean
                    mean, log_var, z = model.encoder.predict([imputed_marker_data, imputed_morph_data])

                    # Create reconstructed date
                    reconstructed_data = pd.DataFrame(columns=markers, data=model.decoder.predict(mean))

                    # Overwrite imputed data with reconstructed data. Drop reconstructed morph data
                    imputed_morph_data = reconstructed_data[
                        ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"]]

                # Derive from ground truth dataset, to only compare imputation values.
                new_imputed_data = ground_truth_data.copy()
                new_imputed_data.loc[imputed_morph_data.index, :] = imputed_morph_data[:]

                # Reconstruct unmodified test data
                encoded_data, reconstructed_data = Predictions.encode_decode_me_vae_data(encoder=model.encoder,
                                                                                         decoder=model.decoder,
                                                                                         data=[ground_truth_marker_data,
                                                                                               ground_truth_morph_data],
                                                                                         markers=markers,
                                                                                         use_mlflow=False)

                reconstructed_r2_scores = reconstructed_r2_scores.append({
                    "Marker": morphological_feature_to_impute,
                    "Score": r2_score(ground_truth_morph_data[morphological_feature_to_impute].iloc[indexes],
                                      reconstructed_data[morphological_feature_to_impute].iloc[indexes]),
                }, ignore_index=True)

                imputed_r2_scores = imputed_r2_scores.append({
                    "Marker": morphological_feature_to_impute,
                    "Score": r2_score(ground_truth_morph_data[morphological_feature_to_impute].iloc[indexes],
                                      imputed_morph_data[morphological_feature_to_impute])
                }, ignore_index=True)

                replaced_r2_scores = replaced_r2_scores.append({
                    "Marker": morphological_feature_to_impute,
                    "Score": r2_score(ground_truth_morph_data[morphological_feature_to_impute].iloc[indexes],
                                      working_morph_data[morphological_feature_to_impute].iloc[indexes])
                }, ignore_index=True)

            # Report results
            plotter: Plotting = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores={"Ground Truth vs. Reconstructed": reconstructed_r2_scores,
                                        "Ground Truth vs. Imputed": imputed_r2_scores,
                                        "Ground Truth vs. Replaced": replaced_r2_scores},
                                file_name=f"R2 score comparison Steps {iter_steps} Percentage {args.percentage}")

            Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=base_path, mlflow_folder="",
                                      prefix="ground_truth")
            Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=base_path, mlflow_folder="",
                                      prefix="imputed")



    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
