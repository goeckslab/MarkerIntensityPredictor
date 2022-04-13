from library.me_vae.me_vae import MEMarkerPredictionVAE
from library.preprocessing.preprocessing import Preprocessing
from library.preprocessing.split import SplitHandler
import pandas as pd


class MEVAEFoldEvaluator:
    @staticmethod
    def evaluate_folds(train_data: pd.DataFrame, amount_of_layers: int, name: str, learning_rate: float = 0.001,
                       embedding_dimension: int = 5) -> list:
        """
        Evaluates the training data using cross fold validation
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
            # Split dataset into marker and morph data
            marker_train_data,  morph_train_data = SplitHandler.split_dataset_into_markers_and_morph_features(train)

            # Normalize the data
            marker_train_data = Preprocessing.normalize(marker_train_data)
            morph_train_data = Preprocessing.normalize(morph_train_data)
            validation = Preprocessing.normalize(validation)

            vae_builder = MEMarkerPredictionVAE()

            model, encoder, decoder, history = vae_builder.build_me_variational_auto_encoder(
                training_data=(marker_train_data, morph_train_data),
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
