import os, sys, time, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score
from library import ExperimentHandler, RunHandler, Replacer, Preprocessing, FolderManagement, Reporter, Plotting, \
    KNNImputation, DataLoader
from typing import List

base_path = "multi_feature_knn_imputation"


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="The name of the experiment which should be evaluated",
                        default="Default", type=str)
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
    run_name: str = "Multi Feature KNN Imputation"

    # Create mlflow tracking client
    experiment_handler: ExperimentHandler = ExperimentHandler(tracking_url=args.tracking_url)
    run_handler: RunHandler = RunHandler(tracking_url=args.tracking_url)

    experiment_name = args.experiment
    associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

    mlflow.set_experiment(experiment_id=associated_experiment_id)

    FolderManagement.create_directory(base_path)

    try:

        if args.spatial:
            run_name: str = f"{run_name} Percentage {args.percentage} Spatial"
        else:
            run_name: str = f"{run_name} Percentage {args.percentage}"

        # Delete previous runs
        run_handler.delete_runs_and_child_runs(experiment_id=associated_experiment_id, run_name=run_name)

        with mlflow.start_run(experiment_id=associated_experiment_id, nested=True,
                              run_name=run_name) as run:
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
                                                               test_data=replaced_test_data_cells,
                                                               missing_values=0)

            print("Calculating r2 scores...")

            score_data: List = []

            for feature in features:
                # Store all cell indexes, to be able to select the correct cells later for r2 comparison
                cell_indexes_to_compare: list = []
                for key, replaced_features in index_replacements.items():
                    if feature in replaced_features:
                        cell_indexes_to_compare.append(key)

                score_data.append({
                    "Marker": feature,
                    "Score": r2_score(test_data[feature].iloc[cell_indexes_to_compare],
                                      imputed_cells[feature].iloc[cell_indexes_to_compare])
                })

            imputed_r2_scores: pd.DataFrame = pd.DataFrame().from_records(score_data)

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
