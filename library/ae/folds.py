import pandas as pd
from library.preprocessing.split import SplitHandler
from library.preprocessing.preprocessing import Preprocessing
from library.ae.auto_encoder import AutoEncoder


class AEFoldEvaluator:
    @staticmethod
    def evaluate_folds(train_data: pd.DataFrame, amount_of_layers: int, name: str, learning_rate: float = 0.001,
                       embedding_dimension: int = 5) -> list:
        evaluation_data: list = []

        model_count: int = 0

        for train_set, validation_set in SplitHandler.create_folds(train_data.copy()):
            print(f"Evaluating fold {model_count}...")
            train_set = Preprocessing.normalize(train_set)
            validation_set = Preprocessing.normalize(validation_set)

            model, encoder, decoder, history = AutoEncoder.build_auto_encoder(training_data=train_set,
                                                                              validation_data=validation_set,
                                                                              input_dimensions=
                                                                              train_set.shape[1],
                                                                              embedding_dimension=embedding_dimension,
                                                                              learning_rate=learning_rate,
                                                                              use_ml_flow=False,
                                                                              amount_of_layers=amount_of_layers)

            evaluation_data.append({"name": f"{name}_{model_count}", "loss": history.history['loss'][-1],
                                    "kl_loss": 0,
                                    "reconstruction_loss": 0,
                                    "learning_rate": learning_rate, "optimizer": "adam",
                                    "model": model, "encoder": encoder, "decoder": decoder,
                                    "amount_of_layers": amount_of_layers, "embedding_dimension": embedding_dimension})
            model_count += 1
            learning_rate += 0.001

        return evaluation_data
