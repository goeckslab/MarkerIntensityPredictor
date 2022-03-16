import argparse
from pathlib import Path
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
import mlflow
from library.mlflow_helper.experiment_handler import ExperimentHandler

base_path = Path("data_imputation")


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


if __name__ == "__main__":
    args = get_args()

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
        cells, markers = DataLoader.load_marker_data(args.file)

        model = mlflow.keras.load_model(f"./mlruns/{model_experiment_id}/{model_run_id}/artifacts/model")
        print(model)



    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
