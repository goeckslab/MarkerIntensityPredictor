import argparse
from library.mlflow_helper.experiment_handler import ExperimentHandler
import mlflow
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
from library.preprocessing.split import create_splits, create_folds
from library.vae.vae import MarkerPredictionVAE
from library.preprocessing.preprocessing import Preprocessing
from library.predictions.predictions import Predictions
import pandas as pd
from sklearn.metrics import r2_score
from library.plotting.plots import Plotting
from pathlib import Path
from library.mlflow_helper.reporter import Reporter

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
    parser.add_argument("--file", action="store", nargs='+', required=True,
                        help="The files used for training the model")
    return parser.parse_args()


def evaluate_folds(train_data: pd.DataFrame, amount_of_layers: int, name: str, learning_rate: float = 0.001) -> list:
    evaluation_data: list = []

    model_count: int = 0

    for train, validation in create_folds(train_data.copy()):
        train = Preprocessing.normalize(train)
        validation = Preprocessing.normalize(validation)

        model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(training_data=train,
                                                                                              validation_data=validation,
                                                                                              input_dimensions=
                                                                                              train.shape[1],
                                                                                              embedding_dimension=5,
                                                                                              learning_rate=learning_rate,
                                                                                              use_ml_flow=False,
                                                                                              amount_of_layers=amount_of_layers)

        evaluation_data.append({"name": f"{name}_{model_count}", "loss": history.history['loss'][-1],
                                "kl_loss": history.history['kl_loss'][-1],
                                "reconstruction_loss":
                                    history.history['reconstruction_loss'][-1],
                                "learning_rate": learning_rate, "optimizer": "adam",
                                "model": model, "encoder": encoder, "decoder": decoder,
                                "amount_of_layers": amount_of_layers})
        model_count += 1

    return evaluation_data


if __name__ == "__main__":
    args = get_args()

    if len(args.file) != 2:
        raise ValueError("Please provide exactly two files. First is the train file, second is the test file")

    train_file = args.file[0]
    test_file = args.file[1]

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

        # Model evaluations are being stored here
        evaluation_data: list = []

        # Load train and test cells
        train_cells, _ = DataLoader.load_marker_data(train_file)
        test_cells, markers = DataLoader.load_marker_data(test_file)

        # Create train test split using the train file data
        train_data, _ = create_splits(cells=train_cells, create_val=False)

        evaluation_data.extend(evaluate_folds(train_data=train_data, amount_of_layers=3, name="Train Data 3"))
        evaluation_data.extend(evaluate_folds(train_data=train_data, amount_of_layers=5, name="Train Data 5"))

        # Create train test split using the test file data
        train_data, _ = create_splits(cells=test_cells, create_val=False)

        evaluation_data.extend(evaluate_folds(train_data=train_data, amount_of_layers=3, name="Test Data 3"))
        evaluation_data.extend(evaluate_folds(train_data=train_data, amount_of_layers=5, name="Test Data 5"))

        # Create combined data of both files
        frames = [train_cells, test_cells]
        combined_data = pd.concat(frames)
        combined_train_data, _ = create_splits(cells=combined_data, create_val=False)

        evaluation_data.extend(
            evaluate_folds(train_data=combined_train_data, amount_of_layers=3, name="Combined Data 3"))
        evaluation_data.extend(
            evaluate_folds(train_data=combined_train_data, amount_of_layers=5, name="Combined Data 5"))

        reconstruction_loss: float = 999999
        selected_fold = {}
        for validation_data in evaluation_data:
            if validation_data["reconstruction_loss"] < reconstruction_loss:
                selected_fold = validation_data
                reconstruction_loss = validation_data["reconstruction_loss"]

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run) as run:
            # Set hyper parameters
            learning_rate: float = float(selected_fold["learning_rate"])
            amount_of_layers: int = selected_fold["amount_of_layers"]
            print(f"Using learning rate {learning_rate}")

            mlflow.log_param("Selected Fold", selected_fold)
            mlflow.log_param("Train File", train_file)
            mlflow.log_param("Test File", test_file)

            # Create train test split for real model training
            train_data, _ = create_splits(cells=train_cells, create_val=False)
            _, test_data = create_splits(cells=test_cells, create_val=False)

            # Normalize
            train_data = Preprocessing.normalize(train_data)
            test_data = Preprocessing.normalize(test_data)

            model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(
                training_data=train_data,
                validation_data=train_data,
                input_dimensions=
                train_data.shape[1],
                embedding_dimension=5,
                learning_rate=learning_rate,
                amount_of_layers=amount_of_layers)

            mean, log_var, z = encoder.predict(test_data)
            encoded_data = pd.DataFrame(z)
            reconstructed_data = pd.DataFrame(columns=markers, data=decoder.predict(encoded_data))

            r2_scores = pd.DataFrame()

            recon_test = pd.DataFrame(data=reconstructed_data, columns=markers)
            ground_truth_data = pd.DataFrame(data=test_data, columns=markers)

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

            plotter = Plotting(base_path=base_path, args=args)

            # Save final model evaluation
            plotter.r2_scores(r2_scores={"VAE": r2_scores}, file_name="r2_score", mlflow_directory="Evaluation")
            Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path, mlflow_folder="Evaluation")

            # Save fold evaluation
            Reporter.report_evaluation(evaluations=evaluation_data, file_name="evaluation_data",
                                       mlflow_folder="Fold Evaluation", save_path=base_path)
            plotter.cross_fold_evaluation(evaluation_data=evaluation_data, value_to_display="loss",
                                          file_name="Loss Distribution", mlflow_folder="Fold Evaluation")
            plotter.cross_fold_evaluation(evaluation_data=evaluation_data, value_to_display="kl_loss",
                                          file_name="KL Loss Distribution", mlflow_folder="Fold Evaluation")
            plotter.cross_fold_evaluation(evaluation_data=evaluation_data, value_to_display="reconstruction_loss",
                                          file_name="Reconstructed Loss Distribution", mlflow_folder="Fold Evaluation")



    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)