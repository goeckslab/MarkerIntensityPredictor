import sys, os
from pathlib import Path
import mlflow
import numpy as np
from typing import List, Dict
import pandas as pd
import time
from tqdm import tqdm
from sklearn.impute import KNNImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from library import DataLoader, FolderManagement, ExperimentHandler, RunHandler, Replacer, Reporter, Preprocessing, \
    KNNImputation, FeatureEngineer
from sklearn.metrics import euclidean_distances, r2_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder

results_folder = Path("knn_distance_imputation")


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
    parser.add_argument("--phenotypes", "-ph", action="store", required=True, help="The phenotype association")
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training the KNN")
    parser.add_argument("--index_replacements", "-ir", action="store", required=False,
                        help="The index replacements with markers to replicate experiments")

    return parser.parse_args()


def get_indexes(x):
    # Drop self nearest index
    x.drop(labels=[x.name], inplace=True)
    return list(x[x].index.values)


def map_indexes_to_phenotype(x, phenotypes: pd.DataFrame):
    phenotypes = [phenotypes.iloc[index].values[0] for index in x]
    return phenotypes


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

    try:

        radius_grid = [50, 150, 300, 450, 600]

        run_name: str = f"KNN Distance Based Data Imputation Percentage {args.percentage}"

        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        cells = DataLoader.load_single_cell_data(file_name=args.file, keep_spatial=True, return_df=True)

        if args.index_replacements is None:
            # Create replacements
            index_replacements: Dict = Replacer.select_index_and_features_to_replace(features=list(cells.columns),
                                                                                     length_of_data=cells.shape[0],
                                                                                     percentage=args.percentage)
        else:
            index_replacements_df: pd.DataFrame = pd.read_csv(args.index_replacements)
            values: List = index_replacements_df.values.tolist()
            index_replacements: Dict = {}
            # Convert dataframe back to expected dictionary
            for i, value in enumerate(values):
                index_replacements[i] = value

        imputed_cell_data: List = []
        replaced_cells_data: List = []
        test_cell_data: List = []

        bulk_engineer: FeatureEngineer = FeatureEngineer(folder=args.folder, file_to_exclude=args.file,
                                                         radius=0)

        test_data_engineer = FeatureEngineer = FeatureEngineer(file=args.file, radius=0)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=run_name) as run:

            # Upload index replacement for data analysis
            Reporter.upload_csv(data=pd.DataFrame.from_dict(index_replacements).T, file_name="index_replacements",
                                save_path=results_folder)

            for radius in radius_grid:
                folder_name: str = f"radius_{radius}"

                bulk_engineer.radius = radius
                print(f"Radius {bulk_engineer.radius}")
                bulk_engineer.start_processing()

                # Report to mlflow
                for key, data in bulk_engineer.results.items():
                    Reporter.upload_csv(data=data, save_path=results_folder, file_name=f"engineered_{key}",
                                        mlflow_folder=folder_name)

                train_data = pd.concat(list(bulk_engineer.results.values()))

                test_data_engineer.radius = 30
                test_data_engineer.start_processing()

                test_data = list(test_data_engineer.results.values())[0]
                # Report to mlflow
                Reporter.upload_csv(data=test_data, save_path=results_folder, file_name=f"test_data_engineered",
                                    mlflow_folder=folder_name)

                replaced_test_data = Replacer.replace_values_by_cell(data=test_data,
                                                                     index_replacements=index_replacements)

                # Report to mlflow
                Reporter.upload_csv(data=replaced_test_data, file_name="replaced_test_data", mlflow_folder=folder_name,
                                    save_path=results_folder)

                columns_to_select = list(set(replaced_test_data.columns) - {"X_centroid", "Y_centroid", "Phenotype",
                                                                            "Cell Neighborhood"})
                print("Imputing data")
                imputer = KNNImputer(n_neighbors=2)
                imputed_cells = KNNImputation.impute(train_data=train_data[columns_to_select],
                                                     test_data=replaced_test_data[columns_to_select],
                                                     missing_values=np.nan)

                Reporter.upload_csv(data=imputed_cells, file_name="imputed_cells", mlflow_folder=folder_name,
                                    save_path=results_folder)

                imputed_cells["Radius"] = radius
                imputed_cell_data.append(imputed_cells)

                replaced_test_data["Radius"] = radius
                replaced_cells_data.append(replaced_test_data)

                test_data["Radius"] = radius
                test_cell_data.append(test_data)

            combined_imputed_cells = pd.concat([data for data in imputed_cell_data])
            combined_replaced_cells = pd.concat([data for data in replaced_cells_data])
            combined_test_cells = pd.concat([data for data in test_cell_data])

            Reporter.upload_csv(data=combined_imputed_cells, file_name="combined_imputed_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_replaced_cells, file_name="combined_replaced_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=combined_test_cells, file_name="combined_test_cells",
                                save_path=results_folder)

            Reporter.upload_csv(data=test_data_engineer.phenotypes[Path(args.file).stem],
                                file_name="test_data_phenotypes",
                                save_path=results_folder)


    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(path=results_folder)
