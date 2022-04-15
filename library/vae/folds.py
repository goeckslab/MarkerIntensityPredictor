import pandas as pd
from library.preprocessing.split import SplitHandler
from library.preprocessing.preprocessing import Preprocessing
from library.vae.vae import MarkerPredictionVAE


class VAEFoldEvaluator:
    @staticmethod
    def evaluate_folds(train_data: pd.DataFrame, amount_of_layers: int, name: str, learning_rate: float = 0.001,
                       embedding_dimension: int = 5) -> list:
        """
        Evaluates models using cross fold validation. Normalization is performed for each split separately.
        @param train_data:
        @param amount_of_layers:
        @param name:
        @param learning_rate:
        @param embedding_dimension:
        @return:
        """
        evaluation_data: list = []

        model_count: int = 0

        for train, validation in SplitHandler.create_folds(train_data.copy()):
            print(f"Evaluating fold {model_count}...")
            train = Preprocessing.normalize(train)
            validation = Preprocessing.normalize(validation)

            model, encoder, decoder, history = MarkerPredictionVAE.build_variational_auto_encoder(training_data=train,
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
            learning_rate += 0.001

        return evaluation_data
