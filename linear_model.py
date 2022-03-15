import mlflow
import argparse
from pathlib import Path
from library.data.data_loader import DataLoader
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.data.folder_management import FolderManagement
from library.mlflow_helper.reporter import Reporter
from library.preprocessing.split import create_splits
from library.preprocessing.preprocessing import Preprocessing
from library.linear.elastic_net import ElasticNet
import pandas as pd


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

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # The file to load ued for data input
    data_file = args.file

    # The base folder
    base_folder = Path(f"linear_{args.run}")

    experiment_name: str = args.experiment
    FolderManagement.create_directory(base_folder)

    try:
        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name=args.experiment)

        with mlflow.start_run(experiment_id=experiment_id, run_name=args.run) as run:
            mlflow.log_param("File", data_file)
            mlflow.log_param("Morphological Data", args.morph)
            mlflow.set_tag("Model", "Linear")
            mlflow.log_param("Seed", args.seed)

            # Load data
            intensities, markers = DataLoader.load_data(file_name=data_file, keep_morph=args.morph)

            Reporter.report_cells_and_markers(save_path=base_folder, cells=intensities, markers=markers)

            # Create train and val from train cells
            train_data, test_data = create_splits(intensities, seed=args.seed, create_val=False)

            # Normalize
            train_data = Preprocessing.normalize(train_data)
            test_data = Preprocessing.normalize(test_data)

            r2_scores: pd.DataFrame = ElasticNet.train_elastic_net(train_data=train_data, test_data=test_data, markers=markers,
                                                           random_state=args.seed, tolerance=0.05)

            print(r2_scores)


    except BaseException as ex:
        print(ex)

    finally:
        print("Cleaning up resources")
        FolderManagement.delete_directory(base_folder)
