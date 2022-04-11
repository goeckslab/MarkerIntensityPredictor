from pathlib import Path
import argparse
from library.knn.knn_imputation import KNNImputation
import pandas as pd
from library.data.folder_management import FolderManagement
import mlflow
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.data.data_loader import DataLoader
from sklearn.metrics import r2_score
from library.preprocessing.preprocessing import Preprocessing
from library.preprocessing.replacements import Replacer
from library.plotting.plots import Plotting
from library.mlflow_helper.reporter import Reporter

# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html


base_path = Path("knn_imputation")


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
    parser.add_argument("--files", action="store", nargs='+', required=True,
                        help="The file used for imputation")
    parser.add_argument("--percentage", "-p", action="store", help="The percentage of data being replaced",
                        default=0.2, required=False, type=float)
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

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
        with mlflow.start_run(experiment_id=associated_experiment_id,
                              run_name=f"{args.run} Percentage {args.percentage}") as run:
            mlflow.log_param("Percentage of replaced values", args.percentage)
            mlflow.log_param("Files", args.files)
            mlflow.log_param("Seed", args.seed)

            if len(args.files) == 1:
                cells, markers = DataLoader.load_marker_data(args.files[0])
            else:
                frames: list = []
                markers = []
                for file in args.files:
                    cells, markers = DataLoader.load_marker_data(file)
                    frames.append(cells)

                cells = pd.concat(frames)
                cells.columns = markers

            normalized_cells = pd.DataFrame(data=Preprocessing.normalize(cells), columns=markers)

            r2_scores: pd.DataFrame = pd.DataFrame()
            for feature in markers:
                print(f"Imputation started for feature {feature} ...")
                # Replace data for specific marker
                working_data, indexes = Replacer.replace_values(normalized_cells.copy(), feature, args.percentage, 0)

                # Select ground truth values for marker and indexes
                selected_ground_truth_values = normalized_cells[feature].iloc[indexes].copy()

                # Impute missing values
                imputed_values = KNNImputation.impute(working_data)

                # Select imputed values for marker and indexes
                selected_imputed_values = imputed_values[feature].iloc[indexes].copy()

                r2_scores = r2_scores.append(
                    {
                        "Marker": feature,
                        "Score": r2_score(selected_ground_truth_values, selected_imputed_values)
                    }, ignore_index=True
                )

            plotter = Plotting(base_path=base_path, args=args)
            plotter.plot_scores(scores={"KNN": r2_scores}, file_name="r2_score")
            Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path)



    except BaseException as ex:
        raise

    finally:
        FolderManagement.delete_directory(base_path)
