import argparse
from pathlib import Path
from library.preprocessing.split import create_splits
import pandas as pd
import numpy as np
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
import mlflow
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.preprocessing.preprocessing import Preprocessing
from sklearn.metrics import r2_score
from library.plotting.plots import Plotting
from typing import Optional

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
    parser.add_argument("--description", "-d", action="store", required=False,
                        help="A description for the experiment to give a broad overview. "
                             "This is only used when a new experiment is being created. Ignored if experiment exists",
                        type=str)
    parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--model", "-m", action="store", nargs="+",
                        help="Specify experiment and run name from where to load the model",
                        type=str, required=True)

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


if __name__ == "__main__":
    args = get_args()
    marker_to_impute = "CK14"

    if len(args.model) != 2:
        raise ValueError("Please specify the experiment as the first parameter and the run name as the second one!")

    # Create mlflow tracking client
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
    experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

    model_experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=args.model[0],
                                                                            create_experiment=False)
    if model_experiment_id is None:
        raise ValueError(f"Could not find experiment {args.model[0]}")

    model_run_id: str = experiment_handler.get_run_id_by_name(experiment_id=model_experiment_id, run_name=args.model[1])

    if model_run_id is None:
        raise ValueError(f"Could not find run with name {args.model[1]}")

    FolderManagement.create_directory(base_path)

    try:
        # load model
        model = mlflow.keras.load_model(f"./mlruns/{model_experiment_id}/{model_run_id}/artifacts/model")

        # Load data
        cells, markers = DataLoader.load_marker_data(args.file)

        train_data, test_data = create_splits(cells=cells, create_val=False, seed=args.seed)

        train_data = pd.DataFrame(columns=markers, data=Preprocessing.normalize(train_data))
        test_data = pd.DataFrame(columns=markers, data=Preprocessing.normalize(test_data))

        ground_truth_data = test_data.copy()

        ground_truth_r2_scores: pd.DataFrame = pd.DataFrame()
        imputed_r2_scores: pd.DataFrame = pd.DataFrame()
        for marker_to_impute in markers:
            # Replace 20% of the data
            test_data[marker_to_impute] = test_data[marker_to_impute].sample(frac=0.8, replace=False)

            index = test_data[test_data[marker_to_impute].isna()].index

            value = np.random.normal(loc=test_data[marker_to_impute].mean(), scale=test_data[marker_to_impute].std(),
                                     size=test_data[marker_to_impute].isna().sum())
            test_data[marker_to_impute].fillna(pd.Series(value, index=index), inplace=True)

            imputed_data: pd.DataFrame = test_data.copy()
            for i in range(7):
                mean, log_var, z = model.encoder.predict(imputed_data)
                encoded_data = pd.DataFrame(z)

                reconstructed_data = pd.DataFrame(columns=markers, data=model.decoder.predict(mean))
                reconstructed_data.loc[index, marker_to_impute] = reconstructed_data[marker_to_impute].mean()
                imputed_data = reconstructed_data

            differences = pd.DataFrame(ground_truth_data[marker_to_impute] - test_data[marker_to_impute])
            differences = differences.loc[(differences != 0).any(1)]
            # print(differences)

            # print("With replacements compared to imputed data")
            # print(r2_score(test_data[marker_to_impute], imputed_data[marker_to_impute]))

            # print("Ground truth data compared to imputed data")
            # print(r2_score(ground_truth_data[marker_to_impute], imputed_data[marker_to_impute]))

            mean, log_var, z = model.encoder.predict(ground_truth_data)
            encoded_data = pd.DataFrame(z)
            reconstructed_data = pd.DataFrame(columns=markers, data=model.decoder.predict(encoded_data))

            # print("Ground truth compared to normal reconstructed data")
            # print(r2_score(ground_truth_data[marker_to_impute], reconstructed_data[marker_to_impute]))
            ground_truth_r2_scores = ground_truth_r2_scores.append({
                "Marker": marker_to_impute,
                "Score": r2_score(ground_truth_data[marker_to_impute], reconstructed_data[marker_to_impute]),
            }, ignore_index=True)

            imputed_r2_scores = imputed_r2_scores.append({
                "Marker": marker_to_impute,
                "Score": r2_score(ground_truth_data[marker_to_impute], imputed_data[marker_to_impute])
            }, ignore_index=True)

        print(ground_truth_r2_scores)
        print(imputed_r2_scores)

        with mlflow.start_run(experiment_id=get_associated_experiment_id(args=args), run_name=args.run) as run:
            plotter: Plotting = Plotting(base_path=base_path, args=args)
            plotter.r2_scores(r2_scores={"Ground Truth": ground_truth_r2_scores, "Imputed": imputed_r2_scores},
                              file_name="r2_scores")

    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
