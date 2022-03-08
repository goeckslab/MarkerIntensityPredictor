from library.data.data_loader import DataLoader
from library.data.folder_management import FolderManagement
from library.mlflow_helper.experiment_handler import ExperimentHandler
import argparse
import mlflow
from pathlib import Path
from library.ae.auto_encoder import AutoEncoder
from library.vae.vae import MarkerPredictionVAE
from library.plotting.plots import Plotting
from library.preprocessing.preprocessing import Preprocessing
from library.mlflow_helper.reporter import Reporter
from library.preprocessing.split import create_splits
from library.evalation.evaluation import Evaluation
from library.predictions.predictions import Predictions

vae_base_result_path = Path("single_biopsy", "VAE")
ae_base_result_path = Path("single_biopsy", "AE")
comparison_base_results_path = Path("single_biopsy")


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
                        type=str)
    parser.add_argument("--description", "-d", action="store", required=False,
                        help="A description for the experiment to give a broad overview. "
                             "This is only used when a new experiment is being created. Ignored if experiment exists",
                        type=str)
    parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--mode", action="store",
                        help="If used only the given model will be executed and no comparison will take place",
                        required=False, choices=['vae', 'ae', 'none'], default="none")

    return parser.parse_args()


def start_ae_experiment(args, experiment_id: str):
    with mlflow.start_run(run_name="AE", nested=True, experiment_id=experiment_id) as run:
        mlflow.log_param("file", args.file)
        mlflow.log_param("morphological_data", args.morph)
        mlflow.set_tag("Model", "AE")

        # Load data
        cells, markers = DataLoader.load_data(file_name=args.file, keep_morph=args.morph)
        Reporter.report_cells_and_markers(save_path=ae_base_result_path, cells=cells, markers=markers)

        normalized_data = Preprocessing.normalize(cells)
        train_data, val_data, test_data = create_splits(normalized_data)

        # Create model
        model, encoder, decoder, history = AutoEncoder.build_auto_encoder(train_data=train_data,
                                                                          validation_data=val_data,
                                                                          input_dimensions=train_data.shape[1],
                                                                          embedding_dimensions=5)

        embeddings, reconstructed_data = Predictions.encode_decode_ae_test_data(encoder=encoder, decoder=decoder,
                                                                                test_data=test_data, markers=markers,
                                                                                save_path=ae_base_result_path,
                                                                                mlflow_directory="AE")

        r2_scores = Evaluation.calculate_r2_score(test_data=test_data, reconstructed_data=reconstructed_data,
                                                  markers=markers)

        # Report r2 score
        Reporter.report_r2_scores(r2_scores=r2_scores, save_path=vae_base_result_path,
                                  mlflow_folder="Evaluation")

        plotter = Plotting(ae_base_result_path, args)
        plotter.plot_model_performance(history, "AE", "Model performance")
        plotter.plot_reconstructed_markers(test_data=test_data, reconstructed_data=reconstructed_data, markers=markers,
                                           mlflow_directory="Evaluation", file_name="Input v Reconstructed")
        plotter.plot_r2_scores(r2_scores, "Evaluation", "R2 scores")
        plotter.plot_markers(train_data=train_data, test_data=test_data,
                             val_data=val_data, markers=markers,
                             mlflow_directory="Evaluation",
                             file_name="Marker Expression")

        # Plot weights
        plotter.plot_weights(encoder.get_layer('encoding_h1').get_weights()[0],
                             markers, "AE", "Encoding layer")
        plotter.plot_weights(decoder.get_layer('decoder_output').get_weights()[0],
                             markers, "AE", "Decoding layer")

        return r2_scores


def start_vae_experiment(args, experiment_id: str):
    # Load cells and markers from the given file
    with mlflow.start_run(run_name="VAE", nested=True, experiment_id=experiment_id) as run:
        mlflow.log_param("file", args.file)
        mlflow.log_param("morphological_data", args.morph)
        mlflow.set_tag("Model", "VAE")

        # Load data
        cells, markers = DataLoader.load_data(file_name=args.file, keep_morph=args.morph)

        normalized_cells = Preprocessing.normalize(cells)
        Reporter.report_cells_and_markers(save_path=vae_base_result_path, cells=cells, markers=markers)

        train_data, val_data, test_data = create_splits(normalized_cells)

        # Create model
        model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(training_data=train_data,
                                                                                              validation_data=val_data,
                                                                                              input_dimensions=
                                                                                              train_data.shape[1],
                                                                                              embedding_dimension=5)

        # Predictions
        encoded_data, reconstructed_data = Predictions.encode_decode_vae_test_data(encoder, decoder, test_data, markers,
                                                                                   save_path=vae_base_result_path,
                                                                                   mlflow_directory="VAE")

        # Evaluate
        evaluation = Evaluation()
        r2_scores = evaluation.calculate_r2_score(test_data=test_data, reconstructed_data=reconstructed_data,
                                                  markers=markers)

        # Report r2 score
        Reporter.report_r2_scores(r2_scores=r2_scores, save_path=vae_base_result_path,
                                  mlflow_folder="Evaluation")

        vae_plotting = Plotting(vae_base_result_path, args)
        vae_plotting.plot_model_performance(model.history, "VAE", "model_performance")
        vae_plotting.plot_reconstructed_markers(test_data=test_data, reconstructed_data=reconstructed_data,
                                                markers=markers, mlflow_directory="Evaluation",
                                                file_name="Initial vs. Reconstructed markers")
        vae_plotting.plot_r2_scores(r2_scores, "Evaluation", "R^2 Scores")
        vae_plotting.plot_markers(train_data=train_data, test_data=test_data,
                                  val_data=val_data, markers=markers,
                                  mlflow_directory="Evaluation",
                                  file_name="Marker Expression")

        vae_plotting.plot_weights(encoder.get_layer('encoding_h1').get_weights()[0],
                                  markers, "VAE", "Encoding layer")
        vae_plotting.plot_weights(decoder.get_layer('decoder_output').get_weights()[0],
                                  markers, "VAE", "Decoding layer")

        return r2_scores


if __name__ == "__main__":
    args = get_args()

    # Create working directories
    FolderManagement.create_directory(vae_base_result_path)
    FolderManagement.create_directory(ae_base_result_path)

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
        mlflow.log_param("File", args.file)
        mlflow.log_param("Mode", args.mode)

        vae_r2_scores = start_vae_experiment(args, experiment_id=associated_experiment_id)
        ae_r2_scores = start_ae_experiment(args, experiment_id=associated_experiment_id)

        # Start experiment which compares AE and VAE
        with mlflow.start_run(run_name="Comparison", nested=True,
                              experiment_id=associated_experiment_id) as comparison:
            print("Comparing vae with ae.")
            plotter = Plotting(comparison_base_results_path, args)
            plotter.plot_r2_scores_comparison(ae_r2_scores=ae_r2_scores,
                                              vae_r2_scores=vae_r2_scores)

    # Cleanup resources
    FolderManagement.delete_directory(ae_base_result_path)
    FolderManagement.delete_directory(vae_base_result_path)