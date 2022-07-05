import os, sys
import mlflow
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library.linear.elastic_net import ElasticNet
from library import DataLoader, Preprocessing, Replacer, RunHandler, ExperimentHandler, FolderManagement, Reporter
from typing import Dict, List
from pathlib import Path
import argparse

results_folder = Path("elastic_net_imputation")


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be used to store the results",
                        default="Default", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training the KNN")
    parser.add_argument("--index_replacements", "-ir", action="store", required=False,
                        help="The index replacements with markers to replicate experiments")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    results_folder = Path(f"{results_folder}_{str(int(time.time_ns() / 1000))}")

    experiment_handler: ExperimentHandler = ExperimentHandler(tracking_url=args.tracking_url)
    run_handler: RunHandler = RunHandler(tracking_url=args.tracking_url)

    experiment_name = args.experiment
    # The id of the associated
    associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                            create_experiment=True)

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(path=results_folder)

    run_name: str = f"Elastic Net Imputation Percentage {args.percentage}"

    run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

    try:
        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:
            test_data = DataLoader.load_single_cell_data(args.file, return_df=True)
            train_data, features, files_used = DataLoader.load_files_in_folder(args.folder, file_to_exclude=args.file)

            test_data = Preprocessing.normalize_new(data=test_data)
            train_data = Preprocessing.normalize_new(data=train_data)

            test_data = test_data[0]
            train_data = train_data[0]

            Reporter.upload_csv(data=test_data, file_name="normalized_test_data", save_path=results_folder)
            Reporter.upload_csv(data=train_data, file_name="normalized_train_data", save_path=results_folder)

            if args.index_replacements is None:
                # Create replacements
                index_replacements: Dict = Replacer.select_index_and_features_to_replace(features=test_data,
                                                                                         length_of_data=test_data.shape[
                                                                                             0],
                                                                                         percentage=args.percentage)
            else:
                index_replacements_df: pd.DataFrame = pd.read_csv(args.index_replacements)
                values: List = index_replacements_df.values.tolist()
                index_replacements: Dict = {}
                # Convert dataframe back to expected dictionary
                for i, value in enumerate(values):
                    index_replacements[i] = value

            replaced_test_data = Replacer.replace_values_by_cell(data=test_data.copy(),
                                                                 index_replacements=index_replacements,
                                                                 value_to_replace=0)

            imputed_data: pd.DataFrame = ElasticNet.impute(train_data=train_data, test_data=replaced_test_data,
                                                           features=test_data.columns,
                                                           random_state=1,
                                                           tolerance=0.01,
                                                           )

            features_to_impute = list(test_data.columns)

            if "X_centroid" in features_to_impute:
                features_to_impute.remove("X_centroid")

            if "Y_centroid" in features_to_impute:
                features_to_impute.remove("Y_centroid")

            Reporter.upload_csv(data=pd.DataFrame.from_dict(index_replacements).T, file_name="index_replacements",
                                save_path=results_folder)
            Reporter.upload_csv(data=replaced_test_data, file_name="replaced_test_data", save_path=results_folder)
            Reporter.upload_csv(data=imputed_data, file_name="imputed_cells", save_path=results_folder)
            Reporter.upload_csv(data=pd.Series(features_to_impute), file_name="features_to_impute",
                                save_path=results_folder)

    except:
        raise

    finally:

        FolderManagement.delete_directory(path=results_folder)
