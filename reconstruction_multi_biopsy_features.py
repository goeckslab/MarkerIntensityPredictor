import pandas as pd
import time
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
from library.me_vae.me_vae import MEMarkerPredictionVAE
from library.me_vae.folds import MEVAEFoldEvaluator
from library.postprocessing.model_selector import ModelSelector


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run", type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="Assigns the run to a particular experiment. "
                             "If the experiment does not exists it will create a new one.",
                        default="Default", type=str)
    parser.add_argument("--file", action="store", nargs='+', required=True, help="The file used for training the model")
    parser.add_argument("--morph", action="store_true", required=False, help="Include morphological data", default=True)
    parser.add_argument("--seed", "-s", action="store", required=False, help="Include morphological data", type=int,
                        default=1)

    return parser.parse_args()


def start_ae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    with mlflow.start_run(run_name="AE", nested=True, experiment_id=experiment_id) as run:
        train_file = args.file[0]
        test_file = args.file[1]

        mlflow.log_param("Train File", train_file)
        mlflow.log_param("Test File", test_file)
        mlflow.log_param("Morphological Data", args.morph)
        mlflow.set_tag("Model", "AE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        train_cells, features = DataLoader.load_single_cell_data(file_name=train_file, keep_morph=args.morph)
        test_cells, features = DataLoader.load_single_cell_data(file_name=test_file, keep_morph=args.morph)

        Reporter.report_cells_and_features(save_path=results_folder, cells=train_cells, features=features,
                                           prefix="train")
        Reporter.report_cells_and_features(save_path=results_folder, cells=test_cells, features=features,
                                           prefix="test")

        train_data = pd.DataFrame(data=train_cells, columns=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            AEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            AEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        # Split train  cells into train and validation data and use for training
        train_data, validation_data = SplitHandler.create_splits(cells=pd.DataFrame(data=train_cells, columns=features),
                                                                 features=features, create_val=False, seed=args.seed)

        # Normalize
        train_data = Preprocessing.normalize(train_data)
        val_data = Preprocessing.normalize(validation_data)

        # Create model
        model, encoder, decoder, history = AutoEncoder.build_auto_encoder(training_data=train_data,
                                                                          validation_data=val_data,
                                                                          input_dimensions=train_data.shape[1],
                                                                          embedding_dimension=5,
                                                                          learning_rate=learning_rate,
                                                                          amount_of_layers=amount_of_layers)

        # Create test data by using the test cells and normalize
        test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

        embeddings, reconstructed_data = Predictions.encode_decode_ae_data(encoder=encoder, decoder=decoder,
                                                                           data=test_data, markers=features,
                                                                           save_path=results_folder,
                                                                           mlflow_directory="Evaluation")

        ae_r2_scores: pd.DataFrame = Evaluation.calculate_r2_scores(ground_truth_data=test_data,
                                                                 compare_data=reconstructed_data,
                                                                 features=features)

        Reporter.report_r2_scores(r2_scores=ae_r2_scores, save_path=Path(results_folder), mlflow_folder="Evaluation")

        plotter = Plotting(results_folder, args)
        plotter.plot_model_architecture(model=encoder, file_name="AE Encoder", mlflow_folder="Evaluation")
        plotter.plot_model_architecture(model=decoder, file_name="AE Decoder", mlflow_folder="Evaluation")
        plotter.plot_model_performance(history, "Evaluation", "Model performance")
        plotter.plot_reconstructed_markers(test_data=test_data, reconstructed_data=reconstructed_data, markers=features,
                                           mlflow_directory="Evaluation", file_name="Input v Reconstructed")
        plotter.plot_scores(scores={"AE": ae_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")
        plotter.plot_feature_intensities(train_data=train_data, test_data=test_data,
                                         val_data=val_data, features=features,
                                         mlflow_directory="Evaluation",
                                         file_name="Marker Expression")

        encoding_h1_weights = encoder.get_layer('encoding_h1').get_weights()[0]
        decoding_output_weights = decoder.get_layer('decoder_output').get_weights()[0]

        Reporter.report_weights(encoding_h1_weights, markers=features, save_path=results_folder,
                                mlflow_folder="Evaluation", file_name="layer_encoding_h1_weights")

        Reporter.report_weights(decoding_output_weights, markers=features, save_path=results_folder,
                                mlflow_folder="Evaluation", file_name="layer_decoding_output_weights")

        # Plot weights
        plotter.plot_weights(weights=encoding_h1_weights, features=features, mlflow_directory="Evaluation",
                             file_name="Encoding layer")
        plotter.plot_weights(weights=decoding_output_weights, features=features, mlflow_directory="Evaluation",
                             file_name="Decoding layer")

        return ae_r2_scores


def start_vae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    # Load cells and markers from the given file
    with mlflow.start_run(run_name="VAE", nested=True, experiment_id=experiment_id) as run:
        train_file = args.file[0]
        test_file = args.file[1]

        mlflow.log_param("Train File", train_file)
        mlflow.log_param("Test File", test_file)
        mlflow.log_param("Morphological Data", args.morph)
        mlflow.set_tag("Model", "VAE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        train_cells, features = DataLoader.load_single_cell_data(file_name=train_file, keep_morph=args.morph)
        test_cells, features = DataLoader.load_single_cell_data(file_name=test_file, keep_morph=args.morph)

        Reporter.report_cells_and_features(save_path=results_folder, cells=train_cells, features=features,
                                           prefix="train")
        Reporter.report_cells_and_features(save_path=results_folder, cells=test_cells, features=features,
                                           prefix="test")

        train_data = pd.DataFrame(data=train_cells, columns=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            VAEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            VAEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        # Split train  cells into train and validation data and use for training
        train_data, validation_data = SplitHandler.create_splits(cells=pd.DataFrame(data=train_cells, columns=features),
                                                                 features=features, create_val=False, seed=args.seed)

        # Normalize
        train_data = Preprocessing.normalize(train_data)
        val_data = Preprocessing.normalize(validation_data)

        # Create model
        model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(training_data=train_data,
                                                                                              validation_data=val_data,
                                                                                              input_dimensions=
                                                                                              train_data.shape[1],
                                                                                              embedding_dimension=5,
                                                                                              learning_rate=learning_rate,
                                                                                              amount_of_layers=amount_of_layers)

        # Create test data by using the test cells
        test_data = pd.DataFrame(data=Preprocessing.normalize(test_cells.copy()), columns=features)

        # Predictions
        embeddings, reconstructed_data = Predictions.encode_decode_vae_data(encoder, decoder, data=test_data,
                                                                            features=features,
                                                                            save_path=results_folder,
                                                                            mlflow_directory="Evaluation")

        # Evaluate
        vae_r2_scores = Evaluation.calculate_r2_scores(ground_truth_data=test_data, compare_data=reconstructed_data,
                                                   features=features)

        Reporter.report_r2_scores(r2_scores=vae_r2_scores, save_path=results_folder,
                                  mlflow_folder="Evaluation")

        vae_plotting = Plotting(results_folder, args)
        vae_plotting.plot_model_architecture(model=encoder, file_name="VAE Encoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_architecture(model=decoder, file_name="VAE Decoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_performance(model.history, "Evaluation", "Model Performance")
        vae_plotting.plot_reconstructed_markers(test_data=test_data, reconstructed_data=reconstructed_data,
                                                markers=features, mlflow_directory="Evaluation",
                                                file_name="Initial vs. Reconstructed markers")
        vae_plotting.plot_scores(scores={"VAE": vae_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")
        vae_plotting.plot_feature_intensities(train_data=train_data, test_data=test_data,
                                              val_data=val_data, features=features,
                                              mlflow_directory="Evaluation",
                                              file_name="Marker Expression")

        encoding_h1_weights = encoder.get_layer('encoding_h1').get_weights()[0]
        decoding_output_weights = decoder.get_layer('decoder_output').get_weights()[0]

        Reporter.report_weights(encoding_h1_weights, markers=features, save_path=results_folder,
                                mlflow_folder="Evaluation", file_name="layer_encoding_h1_weights")

        Reporter.report_weights(decoding_output_weights, markers=features, save_path=results_folder,
                                mlflow_folder="Evaluation", file_name="layer_decoding_output_weights")

        vae_plotting.plot_weights(encoding_h1_weights, features=features, mlflow_directory="Evaluation",
                                  file_name="Encoding layer")
        vae_plotting.plot_weights(decoding_output_weights, features=features, mlflow_directory="Evaluation",
                                  file_name="Decoding layer")

        return vae_r2_scores


def start_me_vae_experiment(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    # Load cells and markers from the given file
    with mlflow.start_run(run_name="ME VAE", nested=True, experiment_id=experiment_id) as run:
        train_file = args.file[0]
        test_file = args.file[1]

        mlflow.log_param("Train File", train_file)
        mlflow.log_param("Test File", test_file)
        mlflow.log_param("Morphological Data", args.morph)
        mlflow.set_tag("Model", "ME VAE")
        mlflow.log_param("Seed", args.seed)

        # Load data
        train_cells, features = DataLoader.load_single_cell_data(file_name=train_file, keep_morph=args.morph)
        test_cells, features = DataLoader.load_single_cell_data(file_name=test_file, keep_morph=args.morph)

        Reporter.report_cells_and_features(save_path=results_folder, cells=train_cells, features=features,
                                           prefix="train")
        Reporter.report_cells_and_features(save_path=results_folder, cells=test_cells, features=features,
                                           prefix="test")

        train_data = pd.DataFrame(data=train_cells, columns=features)

        # Fold evaluation data
        evaluation_data: list = []

        evaluation_data.extend(
            MEVAEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=3, name="3 Layers"))
        evaluation_data.extend(
            MEVAEFoldEvaluator.evaluate_folds(train_data=train_data, amount_of_layers=5, name="5 Layers"))

        # Select to best performing fold
        selected_fold: {} = ModelSelector.select_model_by_lowest_loss(evaluation_data=evaluation_data)

        mlflow.log_param("Selected fold", selected_fold)

        learning_rate: int = selected_fold["learning_rate"]
        amount_of_layers: int = selected_fold["amount_of_layers"]

        # Split train  cells into train and validation data and use for training
        train_data, validation_data = SplitHandler.create_splits(cells=pd.DataFrame(data=train_cells, columns=features),
                                                                 features=features, create_val=False, seed=args.seed)

        marker_train_data, morph_train_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=train_data)

        # Normalize
        marker_train_data = Preprocessing.normalize(marker_train_data)
        morph_train_data = Preprocessing.normalize(morph_train_data)
        validation_data = Preprocessing.normalize(validation_data)

        # Split hold out set into marker and morph data for testing model performance
        marker_test_data, morph_test_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=pd.DataFrame(data=test_cells.copy(), columns=features))

        # Normalize test data
        marker_test_data = pd.DataFrame(data=Preprocessing.normalize(marker_test_data),
                                        columns=marker_test_data.columns)
        morph_test_data = pd.DataFrame(data=Preprocessing.normalize(morph_test_data), columns=morph_test_data.columns)

        test_data = pd.DataFrame(data=Preprocessing.normalize(pd.DataFrame(data=test_cells.copy(), columns=features)),
                                 columns=features)

        # Create model
        model, encoder, decoder, history = MEMarkerPredictionVAE.build_me_variational_auto_encoder(
            training_data=(marker_train_data, morph_train_data),
            validation_data=validation_data,
            input_dimensions=
            train_data.shape[1],
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
        me_vae_r2_scores = Evaluation.calculate_r2_scores(ground_truth_data=test_data, compare_data=reconstructed_data,
                                                   features=features)

        Reporter.report_r2_scores(r2_scores=me_vae_r2_scores, save_path=results_folder,
                                  mlflow_folder="Evaluation")

        vae_plotting = Plotting(results_folder, args)
        vae_plotting.plot_model_architecture(model=encoder, file_name="VAE Encoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_architecture(model=decoder, file_name="VAE Decoder", mlflow_folder="Evaluation")
        vae_plotting.plot_model_performance(model.history, "Evaluation", "Model Performance")
        vae_plotting.plot_reconstructed_markers(test_data=test_data, reconstructed_data=reconstructed_data,
                                                markers=features, mlflow_directory="Evaluation",
                                                file_name="Initial vs. Reconstructed markers")
        vae_plotting.plot_scores(scores={"ME VAE": me_vae_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")
        vae_plotting.plot_feature_intensities(train_data=train_data, test_data=test_data,
                                              val_data=validation_data, features=features,
                                              mlflow_directory="Evaluation",
                                              file_name="Marker Expression")

        return me_vae_r2_scores


def start_elastic_net(args, experiment_id: str, results_folder: Path) -> pd.DataFrame:
    with mlflow.start_run(run_name="ElasticNet", nested=True, experiment_id=experiment_id) as run:
        train_file = args.file[0]
        test_file = args.file[1]

        mlflow.log_param("Train File", train_file)
        mlflow.log_param("Test File", test_file)
        mlflow.log_param("morphological_data", args.morph)
        mlflow.set_tag("Model", "ElasticNet")
        mlflow.log_param("Seed", args.seed)

        # Load data
        train_cells, features = DataLoader.load_single_cell_data(file_name=train_file, keep_morph=args.morph)
        test_cells, features = DataLoader.load_single_cell_data(file_name=test_file, keep_morph=args.morph)

        Reporter.report_cells_and_features(save_path=results_folder, cells=train_cells, features=features,
                                           prefix="train")
        Reporter.report_cells_and_features(save_path=results_folder, cells=test_cells, features=features,
                                           prefix="test")

        # Create train and val from train cells
        train_data, _ = SplitHandler.create_splits(train_cells, seed=args.seed, create_val=False, features=features)
        _, test_data = SplitHandler.create_splits(test_cells, seed=args.seed, create_val=False, features=features)

        # Normalize
        train_data = Preprocessing.normalize(train_data)
        test_data = Preprocessing.normalize(test_data)

        en_r2_scores: pd.DataFrame = ElasticNet.train_elastic_net(train_data=train_data, test_data=test_data,
                                                               features=features,
                                                               random_state=args.seed, tolerance=0.05)

        Reporter.report_r2_scores(r2_scores=en_r2_scores, save_path=results_folder, mlflow_folder="Evaluation")

        plotter = Plotting(results_folder, args)
        plotter.plot_scores(scores={"EN": en_r2_scores}, mlflow_directory="Evaluation", file_name="R2 Scores")

        return en_r2_scores


if __name__ == "__main__":

    args = get_args()

    if len(args.file) != 2:
        raise ValueError(
            "Please specify only 2 files to process! The first one is the train file the second is the test file")

    train_file = args.file[0]
    test_file = args.file[1]

    base_results_path = Path(f"multi_{args.run}_{int(time.time())}")
    ae_base_result_path = Path(base_results_path, "AE")
    vae_base_result_path = Path(base_results_path, "VAE")
    en_base_result_path = Path(base_results_path, "EN")
    me_vae_base_result_path = Path(base_results_path, "ME_VAE")

    # Create base path
    FolderManagement.create_directory(base_results_path)

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
            associated_experiment_id = experiment_handler.get_experiment_id_by_name(experiment_name=experiment_name)

        # Experiment not found
        if associated_experiment_id is None:
            raise ValueError(
                f"Experiment {experiment_name} not found! Either specify a different name or set create_experiment = True.")

        mlflow.set_experiment(experiment_id=associated_experiment_id)

        # Start initial experiment
        with mlflow.start_run(run_name=args.run, nested=True, experiment_id=associated_experiment_id) as run:
            mlflow.log_param("Included Morphological Data", args.morph)
            mlflow.log_param("Train File", train_file)
            mlflow.log_param("Test File", test_file)
            mlflow.log_param("Seed", args.seed)

            en_r2_scores = start_elastic_net(args, experiment_id=associated_experiment_id,
                                             results_folder=en_base_result_path)
            ae_r2_scores = start_ae_experiment(args, experiment_id=associated_experiment_id,
                                               results_folder=ae_base_result_path)

            vae_r2_scores = start_vae_experiment(args, experiment_id=associated_experiment_id,
                                                 results_folder=vae_base_result_path)

            me_vae_r2_scores = start_me_vae_experiment(args=args, experiment_id=associated_experiment_id,
                                                       results_folder=me_vae_base_result_path)

            # Start experiment which compares AE and VAE
            with mlflow.start_run(run_name="Summary", nested=True,
                                  experiment_id=associated_experiment_id) as comparison:
                print("Comparing ml models.")

                # Just load the features
                train_cells, _ = DataLoader.load_single_cell_data(file_name=args.file[0])
                test_cells, features = DataLoader.load_single_cell_data(file_name=args.file[1])

                r2_scores = {"EN": en_r2_scores, "AE": ae_r2_scores, "VAE": vae_r2_scores,
                             "ME VAE": me_vae_r2_scores}

                Reporter.report_r2_scores(r2_scores=vae_r2_scores, save_path=base_results_path, prefix="vae")
                Reporter.report_r2_scores(r2_scores=ae_r2_scores, save_path=base_results_path, prefix="ae")
                Reporter.report_r2_scores(r2_scores=en_r2_scores, save_path=base_results_path, prefix="en")
                Reporter.report_r2_scores(r2_scores=me_vae_r2_scores, save_path=base_results_path, prefix="me_vae")
                # Upload features
                Reporter.upload_csv(data=pd.DataFrame(data=features, columns=["Features"]), save_path=base_results_path,
                                    file_name="Features")

                Reporter.upload_csv(data=train_cells.corr(method='spearman'), save_path=base_results_path,
                                    file_name="train_correlation")

                Reporter.upload_csv(data=test_cells.corr(method='spearman'), save_path=base_results_path,
                                    file_name="test_correlation")

    except BaseException as ex:
        print(ex)
        raise

    finally:
        # Cleanup resources
        FolderManagement.delete_directory(ae_base_result_path)
        FolderManagement.delete_directory(vae_base_result_path)
        FolderManagement.delete_directory(en_base_result_path)
        FolderManagement.delete_directory(base_results_path)
