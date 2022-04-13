import argparse
from pathlib import Path
import mlflow
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from library.data.data_loader import DataLoader
from library.data.folder_management import FolderManagement
from library.mlflow_helper.experiment_handler import ExperimentHandler
from library.mlflow_helper.reporter import Reporter
from library.plotting.plots import Plotting
from library.preprocessing.preprocessing import Preprocessing
from library.preprocessing.split import SplitHandler
from library.me_vae.me_vae import MEMarkerPredictionVAE

base_path = Path("hyper_parameter_tuning")


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
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder used to load the data")
    parser.add_argument("--exclude", action="store", required=False,
                        help="A file which can be excluded from training.", default=None)
    return parser.parse_args()


def evaluate_folds(train_data: pd.DataFrame, amount_of_layers: int, name: str, learning_rate: float = 0.001,
                   embedding_dimension: int = 5) -> list:
    evaluation_data: list = []

    model_count: int = 0

    for train, validation in SplitHandler.create_folds(train_data.copy(), splits=3):
        morph_train_data = train[["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"]]
        marker_train_data = train.drop(["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"], axis=1)
        marker_train_data = Preprocessing.normalize(marker_train_data)
        morph_train_data = Preprocessing.normalize(morph_train_data)
        validation = Preprocessing.normalize(validation)

        # marker_train_data = marker_train_data.reshape(marker_train_data.shape[0], marker_train_data.shape[1])
        # morph_train_data = morph_train_data.reshape(morph_train_data.shape[0], morph_train_data.shape[1])

        vae_builder = MEMarkerPredictionVAE()

        model, encoder, decoder, history = vae_builder.build_me_variational_auto_encoder(
            marker_training_data=marker_train_data,
            morph_training_data=morph_train_data,
            validation_data=validation,
            input_dimensions=
            train.shape[1],
            embedding_dimension=embedding_dimension,
            learning_rate=learning_rate,
            use_ml_flow=False,
            amount_of_layers=amount_of_layers)

        evaluation_data.append({"name": f"{name}_{model_count}", "loss": history.history['loss'][-1],
                                "kl_loss": history.history['kl_loss'][-1],
                                "reconstruction_loss":
                                    history.history['reconstruction_loss'][-1],
                                "learning_rate": learning_rate, "optimizer": "adam",
                                "model": model, "encoder": encoder, "decoder": decoder,
                                "amount_of_layers": amount_of_layers, "embedding_dimension": embedding_dimension})
        model_count += 1

    return evaluation_data


def final_model(data_set: pd.DataFrame, files_used: list, evaluation_duration: float, selected_fold: {}):
    """
    Evaluate the final model
    @param data_set:
    @param files_used:
    @param evaluation_duration:
    @param selected_fold:
    @return:
    """
    learning_rate: float = float(selected_fold["learning_rate"])
    amount_of_layers: int = selected_fold["amount_of_layers"]
    print(f"Using learning rate {learning_rate}")

    mlflow.log_param("Selected Fold", selected_fold)
    mlflow.log_param("Number of Files", len(files_used))
    mlflow.log_param("Files", files_used)
    mlflow.log_param("Evaluation Duration", evaluation_duration)
    mlflow.log_param("Number of cells", data_set.shape[0])
    mlflow.log_param("Number of markers", data_set.shape[1])

    if args.exclude is not None:
        mlflow.log_param("Excluded file", args.exclude)

    original_train_data, original_test_data = SplitHandler.create_splits(cells=data_set,
                                                                         create_val=False)

    # Create copies of original data
    train_data = original_train_data.copy()
    test_data = original_test_data.copy()

    # Split data into marker and morph features
    marker_train_data, morph_train_data = SplitHandler.split_dataset_into_markers_and_morph_features(
        train_data)
    marker_test_data, morph_test_data = SplitHandler.split_dataset_into_markers_and_morph_features(
        test_data)

    marker_columns = marker_train_data.columns
    morph_columns = morph_train_data.columns

    # Normalize
    marker_train_data = Preprocessing.normalize(marker_train_data)
    morph_train_data = Preprocessing.normalize(morph_train_data)

    # Normalize
    marker_test_data = Preprocessing.normalize(marker_test_data)
    morph_test_data = Preprocessing.normalize(morph_test_data)

    vae_builder = MEMarkerPredictionVAE()
    model, encoder, decoder, history = vae_builder.build_me_variational_auto_encoder(
        marker_training_data=marker_train_data,
        morph_training_data=morph_train_data,
        validation_data=train_data,
        input_dimensions=
        train_data.shape[1],
        embedding_dimension=5,
        learning_rate=learning_rate,
        amount_of_layers=amount_of_layers)

    mean, log_var, z = encoder.predict([marker_test_data, morph_test_data])
    encoded_data = pd.DataFrame(z)
    reconstructed_data = pd.DataFrame(columns=markers, data=decoder.predict(encoded_data))

    r2_scores = pd.DataFrame()

    recon_test = pd.DataFrame(data=reconstructed_data, columns=markers)

    # Use the normalized data to create a new ground truth dataset, to achieve real comparison
    frames = [pd.DataFrame(data=marker_test_data, columns=marker_columns),
              pd.DataFrame(data=morph_test_data, columns=morph_columns)]

    ground_truth_data: pd.DataFrame = pd.concat(frames, axis=1)
    ground_truth_data.columns = markers

    for marker in markers:
        ground_truth_marker = ground_truth_data[f"{marker}"]
        reconstructed_marker = recon_test[f"{marker}"]

        score = r2_score(ground_truth_marker, reconstructed_marker)
        r2_scores = r2_scores.append(
            {
                "Marker": marker,
                "Score": score
            }, ignore_index=True
        )

    # Save final model evaluation
    plotter.plot_scores(scores={"ME-VAE": r2_scores}, file_name="r2_score",
                        mlflow_directory="Evaluation")
    Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path, mlflow_folder="Evaluation")

    # Save fold evaluation
    Reporter.report_evaluation(evaluations=evaluation_data, file_name="evaluation_data",
                               mlflow_folder="Fold Evaluation", save_path=base_path)
    plotter.cross_fold_evaluation(evaluation_data=evaluation_data, value_to_display="loss",
                                  file_name="Loss Distribution", mlflow_folder="Fold Evaluation")
    plotter.cross_fold_evaluation(evaluation_data=evaluation_data, value_to_display="kl_loss",
                                  file_name="KL Loss Distribution", mlflow_folder="Fold Evaluation")
    plotter.cross_fold_evaluation(evaluation_data=evaluation_data,
                                  value_to_display="reconstruction_loss",
                                  file_name="Reconstructed Loss Distribution",
                                  mlflow_folder="Fold Evaluation")
    plotter.plot_model_architecture(model=encoder, file_name="Encoder Architecture",
                                    mlflow_folder="Evaluation")
    plotter.plot_model_architecture(model=decoder, file_name="Decoder Architecture",
                                    mlflow_folder="Evaluation")


if __name__ == "__main__":
    # Check whether gpu is available
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU detected. Using gpu training")

    args = get_args()
    # Which files were used
    files_used: list = []

    frames = []
    path_list = Path(args.folder).glob('**/*.csv')
    markers: list = []

    for path in path_list:
        if "SARDANA" in path.stem or args.exclude in path.stem:
            continue

        cells, markers = DataLoader.load_single_cell_data(file_name=str(path))
        frames.append(cells)
        files_used.append(path.stem)

    if len(frames) == 0 or len(markers) == 0:
        raise ValueError("No files found")

    data_set: pd.DataFrame = pd.concat(frames)
    data_set.columns = markers

    evaluation_duration: float = 0

    FolderManagement.create_directory(base_path)
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
                f"Experiment {experiment_name} not found!")

        mlflow.set_experiment(experiment_id=associated_experiment_id)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run) as run:
            plotter = Plotting(base_path=base_path, args=args)

            plotter.plot_correlation(data_set=data_set, file_name="Correlation", mlflow_folder="Evaluation")

            # Model evaluations are being stored here
            evaluation_data: list = []

            # Create train test split using the train file data
            train_data, _ = SplitHandler.create_splits(cells=data_set, create_val=False)

            print("Evaluating data set...")
            start = timer()
            evaluation_data.extend(
                evaluate_folds(train_data=train_data.copy(), amount_of_layers=3, name="Data 3 Embedding 5"))
            evaluation_data.extend(evaluate_folds(train_data=train_data, amount_of_layers=5, name="Data 5 Embedding 5"))
            # evaluation_data.extend(
            #    evaluate_folds(train_data=train_data, amount_of_layers=3, name="Data 3 Embedding 8", embedding_dimension=8))
            # evaluation_data.extend(
            #    evaluate_folds(train_data=train_data, amount_of_layers=5, name="Data 5 Embedding 8", embedding_dimension=8))

            end = timer()
            evaluation_duration = end - start

            # Select best performing model
            reconstruction_loss: float = 999999
            selected_fold = {}
            for validation_data in evaluation_data:
                if validation_data["loss"] < reconstruction_loss:
                    selected_fold = validation_data
                    reconstruction_loss = validation_data["loss"]

            print("Evaluating final model")
            # Final model training
            final_model(data_set=data_set, files_used=files_used, evaluation_duration=evaluation_duration,
                        selected_fold=selected_fold)


    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
