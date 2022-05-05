import mlflow
from pathlib import Path
import time
from library.mlflow_helper.experiment_handler import ExperimentHandler
import argparse
from library.data.data_loader import DataLoader
import pandas as pd
from library.preprocessing.preprocessing import Preprocessing
from library.preprocessing.replacements import Replacer
from library.data.folder_management import FolderManagement
from library.mlflow_helper.reporter import Reporter
from library.plotting.plots import Plotting
from sklearn.metrics import r2_score
from library.knn.knn_imputation import KNNImputation

base_path = "multi_feature_knn_imputation"


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
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training KNN and simple imputer")
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--spatial", action="store_true", help="Include spatial data", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Started knn imputation...")
    base_path = Path(f"{base_path}_{str(int(time.time_ns() / 1000))}")
    run_name: str = "KNN Imputation"

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
            mlflow.log_param("Keep morph", args.morph)
            mlflow.log_param("Keep spatial", args.spatial)

            train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                file_to_exclude=args.file,
                                                                                keep_spatial=args.spatial)
            train_data = pd.DataFrame(data=Preprocessing.normalize(train_cells.copy()), columns=features)

            test_cells, _ = DataLoader.load_single_cell_data(file_name=args.file, keep_spatial=args.spatial)
            test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

            replaced_test_data_cells, index_replacements = Replacer.replace_values_by_cell(data=test_data,
                                                                                           features=features,
                                                                                           percentage=args.percentage)

            print("Imputing data...")
            imputed_cells: pd.DataFrame = KNNImputation.impute(train_data=train_data,
                                                               test_data=replaced_test_data_cells)

            print("Calculating r2 scores...")
            imputed_r2_scores: pd.DataFrame = pd.DataFrame()

            for feature in features:
                # Store all cell indexes, to be able to select the correct cells later for r2 comparison
                cell_indexes_to_compare: list = []
                for key, replaced_features in index_replacements.items():
                    if feature in replaced_features:
                        cell_indexes_to_compare.append(key)

                imputed_r2_scores = imputed_r2_scores.append({
                    "Marker": feature,
                    "Score": r2_score(test_data[feature].iloc[cell_indexes_to_compare],
                                      imputed_cells[feature].iloc[cell_indexes_to_compare])
                }, ignore_index=True)

            plotter = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores={"Imputed": imputed_r2_scores},
                                file_name=f"R2 Imputed Scores {args.percentage}")
            plotter.plot_correlation(data_set=train_cells, file_name="Train Cell Correlation")
            plotter.plot_correlation(data_set=test_cells, file_name="Test Cell Correlation")
            Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=base_path, prefix="imputed")
            Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=base_path,
                                file_name="Features")





    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
