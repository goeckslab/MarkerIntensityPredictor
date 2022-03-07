import pandas as pd
from pathlib import Path
import mlflow
from typing import Tuple


class Predictions:
    @staticmethod
    def encode_decode_vae_test_data(encoder, decoder, test_data: pd.DataFrame, markers: list, save_path: Path,
                                    mlflow_directory: str) -> Tuple:
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """
        mean, log_var, z = encoder.predict(test_data)
        encoded_data = pd.DataFrame(z)
        reconstructed_data = pd.DataFrame(columns=markers, data=decoder.predict(encoded_data))

        encoded_data_save_path = Path(save_path, "encoded_data.csv")
        encoded_data.to_csv(encoded_data_save_path, index=False)
        mlflow.log_artifact(str(encoded_data_save_path), mlflow_directory)

        reconstructed_data_save_path = Path(save_path, "reconstructed_data.csv")
        encoded_data.to_csv(reconstructed_data_save_path, index=False)
        mlflow.log_artifact(str(reconstructed_data_save_path), mlflow_directory)

        return encoded_data, reconstructed_data

    @staticmethod
    def encode_decode_ae_test_data(encoder, decoder, test_data: pd.DataFrame, markers: list, save_path: Path,
                                   mlflow_directory: str) -> Tuple:
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """
        encoded = encoder.predict(test_data)
        encoded_data = pd.DataFrame(encoded)
        reconstructed_data = pd.DataFrame(columns=markers,
                                          data=decoder.predict(encoded_data))

        encoded_data_save_path = Path(save_path, "encoded_data.csv")
        encoded_data.to_csv(encoded_data_save_path, index=False)
        mlflow.log_artifact(str(encoded_data_save_path), mlflow_directory)

        reconstructed_data_save_path = Path(save_path, "reconstructed_data.csv")
        reconstructed_data.to_csv(reconstructed_data_save_path, index=False)
        mlflow.log_artifact(str(reconstructed_data_save_path), mlflow_directory)

        return encoded_data, reconstructed_data
