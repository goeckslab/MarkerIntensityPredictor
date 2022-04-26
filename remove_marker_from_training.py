import argparse
import mlflow
from library.data.data_loader import DataLoader
from library.vae.vae import MarkerPredictionVAE
from library.preprocessing.split import SplitHandler
from library.preprocessing.preprocessing import Preprocessing
import pandas as pd
from library.vae.vae_imputer import VAEImputation
from library.plotting.plots import Plotting
from pathlib import Path
from library.data.folder_management import FolderManagement
from library.mlflow_helper.reporter import Reporter
from library.mlflow_helper.experiment_handler import ExperimentHandler

base_path = Path("remove_marker_from_training")


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
    parser.add_argument("--folder", action="store", required=True, help="The files to use for training")
    parser.add_argument("--exclude", action="store", required=True,
                        help="The file to exclude and use for testing")
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)
    parser.add_argument("--no_mlflow", "-nml", action="store_true", help="Use ml flow?", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    use_mlflow = not args.no_mlflow

    base_path = Path(f"{base_path}_{args.run}")
    FolderManagement.create_directory(path=base_path)

    try:

        # load train data
        train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                            file_to_exclude=args.exclude)
        # Load test data
        test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)
        test_cells = pd.DataFrame(data=test_cells, columns=features)
        test_cells = pd.DataFrame(data=Preprocessing.normalize(data=test_cells), columns=features)

        # save ground truth
        ground_truth_test: pd.DataFrame = pd.DataFrame(data=test_cells.copy(), columns=features)

        train_cells, validation_cells = SplitHandler.create_splits(cells=train_cells, create_val=False,
                                                                   features=features)

        train_cells = pd.DataFrame(data=train_cells, columns=features)
        validation_cells = pd.DataFrame(data=validation_cells, columns=features)

        imputed_r2_scores: pd.DataFrame = pd.DataFrame()
        reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()
        replaced_r2_scores: pd.DataFrame = pd.DataFrame()

        previous_feature: str = None

        for feature in features:
            # Make copy
            working_train_cells = train_cells.copy()
            working_validation_cells = validation_cells.copy()

            # Set feature to 0
            working_train_cells[feature].values[:] = 0
            working_validation_cells[feature].values[:] = 0

            if previous_feature is not None:
                assert not working_validation_cells[previous_feature].eq(
                    0.0).any(), f"Feature {previous_feature} should not contain any 0 "
                assert not working_train_cells[previous_feature].eq(
                    0.0).any(), f"Feature {previous_feature} should not contain any 0 "

            assert working_validation_cells[feature].eq(0.0).all(), f"Feature {feature} should only contain 0 "
            assert working_train_cells[feature].eq(0.0).all(), f"Feature {feature} should only contain 0 "

            working_train_cells = pd.DataFrame(data=Preprocessing.normalize(data=working_train_cells),
                                               columns=features)
            working_validation_cells = pd.DataFrame(data=Preprocessing.normalize(data=working_validation_cells),
                                                    columns=features)

            model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(
                training_data=working_train_cells,
                validation_data=working_validation_cells,
                input_dimensions=
                working_train_cells.shape[1],
                embedding_dimension=5,
                use_ml_flow=False)

            imputed_r2_score, reconstructed_r2_score, replaced_r2_score = VAEImputation.impute_data_by_feature(
                model=model,
                iter_steps=1,
                ground_truth_data=ground_truth_test,
                feature_to_impute=feature,
                percentage=1,
                features=features)

            imputed_r2_scores = imputed_r2_scores.append(
                {
                    "Marker": feature,
                    "Score": imputed_r2_score
                }, ignore_index=True)

            reconstructed_r2_scores = reconstructed_r2_scores.append(
                {
                    "Marker": feature,
                    "Score": reconstructed_r2_score
                }, ignore_index=True)

            replaced_r2_scores = replaced_r2_scores.append(
                {
                    "Marker": feature,
                    "Score": replaced_r2_score
                }, ignore_index=True)

            previous_feature = feature

        if use_mlflow:
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
                    f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

            mlflow.set_experiment(experiment_id=associated_experiment_id)

            with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run) as run:
                plotter: Plotting = Plotting(args=args, base_path=base_path)

                plotter.plot_scores({"Imputed": imputed_r2_scores}, file_name="Imputed Features")
                plotter.plot_scores({"Reconstructed": reconstructed_r2_scores}, file_name="Reconstructed Features")
                plotter.plot_scores({"Replaced": replaced_r2_scores}, file_name="Replaced Features")

                Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=base_path, prefix="imputed")
                Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=base_path,
                                          prefix="reconstructed")
                Reporter.report_r2_scores(r2_scores=replaced_r2_scores, save_path=base_path, prefix="replaced")

        else:
            plotter: Plotting = Plotting(args=args, base_path=base_path)

            plotter.plot_scores({"Imputed": imputed_r2_scores}, file_name="Imputed Features", use_mlflow=False)
            plotter.plot_scores({"Reconstructed": reconstructed_r2_scores}, file_name="Reconstructed Features",
                                use_mlflow=False)
            plotter.plot_scores({"Replaced": replaced_r2_scores}, file_name="Replaced Features", use_mlflow=False)

            Reporter.report_r2_scores(r2_scores=imputed_r2_scores, save_path=base_path, prefix="imputed",
                                      use_mlflow=False)
            Reporter.report_r2_scores(r2_scores=reconstructed_r2_scores, save_path=base_path,
                                      prefix="reconstructed", use_mlflow=False)
            Reporter.report_r2_scores(r2_scores=replaced_r2_scores, save_path=base_path, prefix="replaced",
                                      use_mlflow=False)




    except:
        FolderManagement.delete_directory(path=base_path)
        raise

    finally:
        if use_mlflow:
            FolderManagement.delete_directory(path=base_path)
