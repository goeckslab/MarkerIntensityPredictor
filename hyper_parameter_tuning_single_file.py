import argparse
from library.mlflow_helper.experiment_handler import ExperimentHandler
import mlflow
from library.data.folder_management import FolderManagement
from library.data.data_loader import DataLoader
from library.preprocessing.split import SplitHandler
from library.vae.vae import MarkerPredictionVAE
from library.preprocessing.preprocessing import Preprocessing
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
    parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
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

        cells, markers = DataLoader.load_marker_data(args.file)
        train_data, test_data = SplitHandler.create_splits(cells=cells, create_val=False)

        eval_data: list = []
        model_count: int = 0
        learning_rate: float = 0.001
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
                                                                                                  amount_of_layers=3)

            eval_data.append({"name": f"3_{model_count}", "loss": history.history['loss'][-1],
                              "kl_loss": history.history['kl_loss'][-1],
                              "reconstruction_loss": history.history['reconstruction_loss'][-1],
                              "learning_rate": learning_rate, "optimizer": "adam",
                              "model": model, "encoder": encoder, "decoder": decoder,
                              "amount_of_layers": 3})

            learning_rate += 0.002
            model_count += 1

        model_count: int = 0
        learning_rate: float = 0.001
        for train, validation in SplitHandler.create_folds(train_data.copy()):
            train = Preprocessing.normalize(train)
            validation = Preprocessing.normalize(validation)

            model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(training_data=train,
                                                                                                  validation_data=validation,
                                                                                                  input_dimensions=
                                                                                                  train.shape[1],
                                                                                                  embedding_dimension=5,
                                                                                                  learning_rate=learning_rate,
                                                                                                  use_ml_flow=False,
                                                                                                  amount_of_layers=5)

            eval_data.append({"name": f"5_{model_count}", "loss": history.history['loss'][-1],
                              "kl_loss": history.history['kl_loss'][-1],
                              "reconstruction_loss": history.history['reconstruction_loss'][-1],
                              "learning_rate": learning_rate, "optimizer": "adam",
                              "model": model, "encoder": encoder, "decoder": decoder,
                              "amount_of_layers": 5})

            learning_rate += 0.002
            model_count += 1

        reconstruction_loss: float = 999999
        selected_fold = {}
        for evaluation in eval_data:
            if evaluation["loss"] < reconstruction_loss:
                selected_fold = evaluation
                reconstruction_loss = evaluation["loss"]

        # Normalize
        train_data = Preprocessing.normalize(train_data)
        test_data = Preprocessing.normalize(test_data)

        with mlflow.start_run(experiment_id=associated_experiment_id, run_name=args.run) as run:
            # Set hyper parameters
            learning_rate = float(selected_fold["learning_rate"])
            print(f"Using learning rate {learning_rate}")

            mlflow.log_param("Selected Fold", selected_fold)
            mlflow.log_param("File", args.file)

            model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(
                training_data=train_data,
                validation_data=train_data,
                input_dimensions=
                train_data.shape[1],
                embedding_dimension=5,
                learning_rate=learning_rate)

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
            plotter.plot_scores(scores={"VAE": r2_scores}, file_name="r2_score", mlflow_directory="Evaluation")
            Reporter.report_r2_scores(r2_scores=r2_scores, save_path=base_path, mlflow_folder="Evaluation")

    except:
        raise
    finally:
        FolderManagement.delete_directory(base_path)
