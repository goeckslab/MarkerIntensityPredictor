from library.data.data_loader import DataLoader
from library.data.folder_management import FolderManagement
from library.mlflow_helper.experiment_handler import ExperimentHandler
import argparse
import mlflow
from pathlib import Path
from library.ae.auto_encoder import AutoEncoder
from library.ae.folds import AEFoldEvaluator
from library.vae.vae import MarkerPredictionVAE
from library.vae.folds import VAEFoldEvaluator
from library.plotting.plots import Plotting
from library.preprocessing.preprocessing import Preprocessing
from library.mlflow_helper.reporter import Reporter
from library.preprocessing.split import SplitHandler
from library.evalation.evaluation import Evaluation
from library.predictions.predictions import Predictions
from library.linear.elastic_net import ElasticNet
import pandas as pd
import time
from library.postprocessing.model_selector import ModelSelector
from library.me_vae.me_vae import MEMarkerPredictionVAE
from library.me_vae.folds import MEVAEFoldEvaluator


# normalizing https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data

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
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder used to load the data")
    parser.add_argument("--exclude", action="store", required=False,
                        help="A file which can be excluded from training.", default=None)
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)

    return parser.parse_args()


def start_ae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    with mlflow.start_run(run_name="AE", nested=True, experiment_id=experiment_id) as run:
        mlflow.log_param("Use morphological features", args.morph)
        mlflow.set_tag("Model", "AE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        ae_train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                               file_to_exclude=args.exclude)

        # Report files used
        mlflow.log_param("Number of Files", len(files_used))
        mlflow.log_param("Files", files_used)
        mlflow.log_param("Excluded File", args.exclude)

        Reporter.report_cells_and_features(save_path=results_folder, cells=ae_train_cells, features=features)

        ae_train_data, ae_validation_data = SplitHandler.create_splits(cells=ae_train_cells, create_val=False,
                                                                       seed=args.seed,
                                                                       features=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            AEFoldEvaluator.evaluate_folds(train_data=ae_train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            AEFoldEvaluator.evaluate_folds(train_data=ae_train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        # Normalize train and validation data
        ae_train_data = pd.DataFrame(data=Preprocessing.normalize(ae_train_data), columns=features)
        ae_validation_data = pd.DataFrame(data=Preprocessing.normalize(ae_validation_data), columns=features)

        # Load and normalize test data
        ae_test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)
        ae_test_cells: pd.DataFrame = pd.DataFrame(data=ae_test_cells, columns=features)

        ae_test_data = pd.DataFrame(data=Preprocessing.normalize(ae_test_cells), columns=features)

        # Create model
        model, encoder, decoder, history = AutoEncoder.build_auto_encoder(training_data=ae_train_data,
                                                                          validation_data=ae_validation_data,
                                                                          input_dimensions=ae_train_data.shape[1],
                                                                          embedding_dimension=5,
                                                                          learning_rate=learning_rate,
                                                                          amount_of_layers=amount_of_layers)

        embeddings, reconstructed_data = Predictions.encode_decode_ae_data(encoder=encoder, decoder=decoder,
                                                                           data=ae_test_data, markers=features,
                                                                           save_path=results_folder,
                                                                           mlflow_directory="Evaluation")

        ae_r2_scores = Evaluation.calculate_r2_scores(ground_truth_data=ae_test_data, compare_data=reconstructed_data,
                                                      features=features)

        # Report r2 score
        Reporter.report_r2_scores(r2_scores=ae_r2_scores, save_path=vae_base_result_path,
                                  mlflow_folder="Evaluation")

        plotter = Plotting(results_folder, args)
        plotter.plot_model_architecture(model=encoder, file_name="AE Encoder", mlflow_folder="Evaluation")
        plotter.plot_model_architecture(model=decoder, file_name="AE Decoder", mlflow_folder="Evaluation")
        plotter.plot_model_performance(history=history, file_name="Model performance", mlflow_directory="Evaluation")
        plotter.plot_reconstructed_markers(test_data=ae_test_data, reconstructed_data=reconstructed_data,
                                           markers=features,
                                           mlflow_directory="Evaluation", file_name="Input v Reconstructed")
        plotter.plot_scores(scores={"AE": ae_r2_scores}, mlflow_directory="Evaluation", file_name="R2 scores")
        plotter.plot_feature_intensities(train_data=ae_train_data, test_data=ae_test_data,
                                         val_data=ae_validation_data, features=features,
                                         mlflow_directory="Evaluation",
                                         file_name="Marker Expression")

        plotter.plot_correlation(data_set=ae_train_cells, file_name="Train Cells Correlation",
                                 mlflow_folder="Evaluation")
        plotter.plot_correlation(data_set=ae_test_cells, file_name="Test Cells Correlation", mlflow_folder="Evaluation")
        return ae_r2_scores


def start_vae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    # Load cells and markers from the given file
    with mlflow.start_run(run_name="VAE", nested=True, experiment_id=experiment_id) as run:
        mlflow.log_param("Use morphological features", args.morph)
        mlflow.set_tag("Model", "VAE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        vae_train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                file_to_exclude=args.exclude)

        # Report files used
        mlflow.log_param("Number of Files", len(files_used))
        mlflow.log_param("Files", files_used)
        mlflow.log_param("Excluded File", args.exclude)

        Reporter.report_cells_and_features(save_path=results_folder, cells=vae_train_cells, features=features)

        vae_train_data, vae_validation_data = SplitHandler.create_splits(cells=vae_train_cells.copy(), create_val=False,
                                                                         seed=args.seed,
                                                                         features=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            VAEFoldEvaluator.evaluate_folds(train_data=vae_train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            VAEFoldEvaluator.evaluate_folds(train_data=vae_train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        # Normalize train and validation
        vae_train_data = pd.DataFrame(data=Preprocessing.normalize(vae_train_data.copy()), columns=features)
        vae_validation_data = pd.DataFrame(data=Preprocessing.normalize(vae_validation_data.copy()), columns=features)

        # Load and normalize test data
        vae_test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)
        vae_test_cells = pd.DataFrame(data=vae_test_cells, columns=features)

        vae_test_data = pd.DataFrame(data=Preprocessing.normalize(vae_test_cells.copy()), columns=features)

        # Create model
        model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(
            training_data=vae_train_data,
            validation_data=vae_validation_data,
            input_dimensions=
            vae_train_data.shape[1],
            embedding_dimension=5,
            learning_rate=learning_rate,
            amount_of_layers=amount_of_layers,
            use_ml_flow=True)

        # Predictions
        encoded_data, reconstructed_data = Predictions.encode_decode_vae_data(encoder, decoder, data=vae_test_data,
                                                                              features=features,
                                                                              save_path=results_folder,
                                                                              mlflow_directory="Evaluation")

        # Evaluate
        vae_r2_scores: pd.DataFrame = Evaluation.calculate_r2_scores(ground_truth_data=vae_test_data,
                                                                     compare_data=reconstructed_data,
                                                                     features=features)

        # Report r2 score
        Reporter.report_r2_scores(r2_scores=vae_r2_scores, save_path=results_folder,
                                  mlflow_folder="Evaluation")

        vae_plotting = Plotting(results_folder, args)
        vae_plotting.plot_model_architecture(model=encoder, file_name="VAE Encoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_architecture(model=decoder, file_name="VAE Decoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_performance(history=model.history, mlflow_directory="Evaluation",
                                            file_name="Model Performance")
        vae_plotting.plot_reconstructed_markers(test_data=vae_test_data, reconstructed_data=reconstructed_data,
                                                markers=features, mlflow_directory="Evaluation",
                                                file_name="Initial vs. Reconstructed markers")
        vae_plotting.plot_scores(scores={"VAE": vae_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")
        vae_plotting.plot_feature_intensities(train_data=vae_train_data, test_data=vae_test_data,
                                              val_data=vae_validation_data, features=features,
                                              mlflow_directory="Evaluation",
                                              file_name="Marker Expression")
        vae_plotting.plot_correlation(data_set=vae_train_cells, file_name="Train Cells Correlation",
                                      mlflow_folder="Evaluation")
        vae_plotting.plot_correlation(data_set=vae_test_cells, file_name="Test Cells Correlation",
                                      mlflow_folder="Evaluation")

        return vae_r2_scores


def start_me_vae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    # Load cells and markers from the given file
    with mlflow.start_run(run_name="ME VAE", nested=True, experiment_id=experiment_id) as run:
        mlflow.log_param("Use morphological features", args.morph)
        mlflow.set_tag("Model", "ME VAE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        me_vae_train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                                   file_to_exclude=args.exclude)

        # Report files used
        mlflow.log_param("Number of Files", len(files_used))
        mlflow.log_param("Files", files_used)
        mlflow.log_param("Excluded File", args.exclude)

        Reporter.report_cells_and_features(save_path=results_folder, cells=me_vae_train_cells, features=features)

        me_vae_train_data, me_vae_validation_data = SplitHandler.create_splits(cells=me_vae_train_cells,
                                                                               create_val=False, seed=args.seed,
                                                                               features=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            MEVAEFoldEvaluator.evaluate_folds(train_data=me_vae_train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            MEVAEFoldEvaluator.evaluate_folds(train_data=me_vae_train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        me_vae_marker_train_data, me_vae_morph_train_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=me_vae_train_data)

        me_vae_marker_validation_data, me_vae_morph_validation_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=me_vae_validation_data)

        # Normalize
        me_vae_marker_train_data: pd.DataFrame = pd.DataFrame(
            data=Preprocessing.normalize(me_vae_marker_train_data.copy()),
            columns=me_vae_marker_train_data.columns)

        me_vae_morph_train_data: pd.DataFrame = pd.DataFrame(
            data=Preprocessing.normalize(me_vae_morph_train_data.copy()),
            columns=me_vae_morph_train_data.columns)

        me_vae_marker_validation_data: pd.DataFrame = pd.DataFrame(
            data=Preprocessing.normalize(me_vae_marker_validation_data.copy()),
            columns=me_vae_marker_validation_data.columns)

        me_vae_morph_validation_data: pd.DataFrame = pd.DataFrame(
            data=Preprocessing.normalize(me_vae_morph_validation_data.copy()),
            columns=me_vae_morph_validation_data.columns)

        # Load test cell, which is the excluded data file
        me_vae_test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)

        me_vae_test_cells = pd.DataFrame(data=me_vae_test_cells, columns=features)

        # Split hold out set into marker and morph data for testing model performance
        marker_test_data, morph_test_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=me_vae_test_cells.copy())

        # Normalize test data
        marker_test_data = pd.DataFrame(data=Preprocessing.normalize(marker_test_data),
                                        columns=marker_test_data.columns)
        morph_test_data = pd.DataFrame(data=Preprocessing.normalize(morph_test_data), columns=morph_test_data.columns)

        me_vae_test_data = pd.DataFrame(data=Preprocessing.normalize(me_vae_test_cells.copy()), columns=features)

        # Create model
        model, encoder, decoder, history = MEMarkerPredictionVAE.build_me_variational_auto_encoder(
            training_data=(me_vae_marker_train_data, me_vae_morph_train_data),
            validation_data=(me_vae_marker_validation_data, me_vae_morph_validation_data),
            output_dimensions=len(features),
            embedding_dimension=5,
            learning_rate=learning_rate,
            amount_of_layers=amount_of_layers)

        # Predictions
        encoded_data, reconstructed_data = Predictions.encode_decode_me_vae_data(encoder, decoder,
                                                                                 data=[marker_test_data,
                                                                                       morph_test_data],
                                                                                 features=features,
                                                                                 save_path=results_folder,
                                                                                 mlflow_directory="Evaluation")

        # Evaluate
        me_vae_r2_scores = Evaluation.calculate_r2_scores(ground_truth_data=me_vae_test_data,
                                                          compare_data=reconstructed_data,
                                                          features=features)

        # Report r2 score
        Reporter.report_r2_scores(r2_scores=me_vae_r2_scores, save_path=results_folder,
                                  mlflow_folder="Evaluation")

        vae_plotting = Plotting(results_folder, args)
        vae_plotting.plot_model_architecture(model=encoder, file_name="ME VAE Encoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_architecture(model=decoder, file_name="ME VAE Decoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_performance(history=model.history, mlflow_directory="Evaluation",
                                            file_name="Model Performance")
        vae_plotting.plot_reconstructed_markers(test_data=me_vae_test_data, reconstructed_data=reconstructed_data,
                                                markers=features, mlflow_directory="Evaluation",
                                                file_name="Initial vs. Reconstructed markers")
        vae_plotting.plot_scores(scores={"ME VAE": me_vae_r2_scores}, mlflow_directory="Evaluation",
                                 file_name="R2 Scores")
        vae_plotting.plot_feature_intensities(train_data=me_vae_train_data, test_data=me_vae_test_data,
                                              val_data=me_vae_validation_data, features=features,
                                              mlflow_directory="Evaluation",
                                              file_name="Marker Expression")
        vae_plotting.plot_correlation(data_set=me_vae_train_cells, file_name="Train Cells Correlation",
                                      mlflow_folder="Evaluation")
        vae_plotting.plot_correlation(data_set=me_vae_test_cells, file_name="Test Cells Correlation",
                                      mlflow_folder="Evaluation")

        return me_vae_r2_scores


def start_elastic_net(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    with mlflow.start_run(run_name="ElasticNet", nested=True, experiment_id=experiment_id) as run:
        print("Evaluating EN model...")
        mlflow.log_param("Use morphological features", args.morph)
        mlflow.set_tag("Model", "ElasticNet")
        mlflow.log_param("Seed", args.seed)

        # Load train data
        train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                            file_to_exclude=args.exclude)
        test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)

        # Report files used
        mlflow.log_param("Number of Files", len(files_used))
        mlflow.log_param("Files", files_used)
        mlflow.log_param("Excluded File", args.exclude)

        Reporter.report_cells_and_features(save_path=results_folder, cells=train_cells, features=features)

        # Normalize train and test data
        train_data = pd.DataFrame(data=Preprocessing.normalize(train_cells.copy()), columns=features)
        test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

        en_r2_scores: pd.DataFrame = ElasticNet.train_elastic_net(train_data=train_data, test_data=test_data,
                                                                  features=features,
                                                                  random_state=args.seed, tolerance=0.05)

        Reporter.report_r2_scores(r2_scores=en_r2_scores, save_path=results_folder, mlflow_folder="Evaluation")

        plotter = Plotting(results_folder, args)
        plotter.plot_scores(scores={"EN": en_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")
        plotter.plot_correlation(data_set=train_cells, file_name="Train Cells Correlation", mlflow_folder="Evaluation")
        plotter.plot_correlation(data_set=test_cells, file_name="Test Cells Correlation", mlflow_folder="Evaluation")
        return en_r2_scores


if __name__ == "__main__":
    args = get_args()
    base_results_path = Path(f"in_sample_{args.run}_{int(time.time())}")
    vae_base_result_path = Path(base_results_path, "VAE")
    ae_base_result_path = Path(base_results_path, "AE")
    en_base_result_path = Path(base_results_path, "EN")
    me_vae_base_result_path = Path(base_results_path, "ME_VAE")

    # Create working directories
    FolderManagement.create_directory(vae_base_result_path)
    FolderManagement.create_directory(ae_base_result_path)
    FolderManagement.create_directory(en_base_result_path)
    FolderManagement.create_directory(me_vae_base_result_path)

    try:
        # set tracking url
        if args.tracking_url is not None:
            mlflow.set_tracking_uri(args.tracking_url)

        # Create mlflow tracking client
        client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)
        experiment_handler: ExperimentHandler = ExperimentHandler(client=client)

        # The id of the associated
        associated_experiment_id = None

        experiment_name = args.experiment
        if experiment_name is not None:
            associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name,
                                                                                    experiment_description=args.description)

        # Experiment not found
        if associated_experiment_id is None:
            raise ValueError(
                f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

        mlflow.set_experiment(experiment_id=associated_experiment_id)

        # Start initial experiment
        with mlflow.start_run(run_name=args.run, nested=True, experiment_id=associated_experiment_id) as run:
            mlflow.log_param("Included Morphological Data", args.morph)
            mlflow.log_param("Seed", args.seed)

            me_vae_r2_scores = start_me_vae_experiment(args=args, experiment_id=associated_experiment_id,
                                                       results_folder=me_vae_base_result_path)

            ae_r2_scores = start_ae_experiment(args, experiment_id=associated_experiment_id,
                                               results_folder=ae_base_result_path)

            vae_r2_scores = start_vae_experiment(args, experiment_id=associated_experiment_id,
                                                 results_folder=vae_base_result_path)

            en_r2_scores = start_elastic_net(args, experiment_id=associated_experiment_id,
                                             results_folder=en_base_result_path)

            # Start experiment which compares the ml models
            with mlflow.start_run(run_name="Summary", nested=True,
                                  experiment_id=associated_experiment_id) as comparison:
                print("Comparing ml models...")

                excluded_cells, features = DataLoader.load_single_cell_data(file_name=args.exclude)

                used_cells, _, _ = DataLoader.load_files_in_folder(folder=args.folder, file_to_exclude=args.exclude)

                r2_scores = {"EN": en_r2_scores, "AE": ae_r2_scores, "VAE": vae_r2_scores,
                             "ME VAE": me_vae_r2_scores}

                Reporter.report_r2_scores(r2_scores=vae_r2_scores, save_path=base_results_path, prefix="vae")
                Reporter.report_r2_scores(r2_scores=ae_r2_scores, save_path=base_results_path, prefix="ae")
                Reporter.report_r2_scores(r2_scores=en_r2_scores, save_path=base_results_path, prefix="en")
                Reporter.report_r2_scores(r2_scores=me_vae_r2_scores, save_path=base_results_path, prefix="me_vae")
                # Upload features
                Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=base_results_path,
                                    file_name="Features")
                Reporter.upload_csv(data=excluded_cells.corr(method='spearman'), save_path=base_results_path,
                                    file_name="excluded_cells_correlation")
                Reporter.upload_csv(data=used_cells.corr(method='spearman'), save_path=base_results_path,
                                    file_name="used_cells_correlation")


    except BaseException as ex:
        raise
    finally:
        # Cleanup resources
        FolderManagement.delete_directory(base_results_path)
