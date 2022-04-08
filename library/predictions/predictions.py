import pandas as pd
from pathlib import Path
import mlflow
from typing import Tuple


class Predictions:
    @staticmethod
    def encode_decode_vae_data(encoder, decoder, data: pd.DataFrame, markers: list, save_path: Path = None,
                               mlflow_directory: str = None, use_mlflow: bool = True) -> Tuple:
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """

        if use_mlflow:
            if mlflow_directory is None or save_path is None:
                raise ValueError("When use_mlflow is true, save_path and mlflow_directory are mandatory")

        mean, log_var, z = encoder.predict(data)
        encoded_data = pd.DataFrame(z)
        reconstructed_data = pd.DataFrame(columns=markers, data=decoder.predict(encoded_data))

        if use_mlflow:
            encoded_data_save_path = Path(save_path, "encoded_data.csv")
            encoded_data.to_csv(encoded_data_save_path, index=False)
            mlflow.log_artifact(str(encoded_data_save_path), mlflow_directory)

            reconstructed_data_save_path = Path(save_path, "reconstructed_data.csv")
            encoded_data.to_csv(reconstructed_data_save_path, index=False)
            mlflow.log_artifact(str(reconstructed_data_save_path), mlflow_directory)

        return encoded_data, reconstructed_data

    @staticmethod
    def encode_decode_me_vae_data(encoder, decoder, data: list, markers: list, save_path: Path = None,
                                  mlflow_directory: str = None, use_mlflow: bool = True) -> Tuple:
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """

        if use_mlflow:
            if mlflow_directory is None or save_path is None:
                raise ValueError("When use_mlflow is true, save_path and mlflow_directory are mandatory")

        mean, log_var, z = encoder.predict(data)
        encoded_data = pd.DataFrame(z)
        reconstructed_data = pd.DataFrame(columns=markers, data=decoder.predict(encoded_data))

        if use_mlflow:
            encoded_data_save_path = Path(save_path, "encoded_data.csv")
            encoded_data.to_csv(encoded_data_save_path, index=False)
            mlflow.log_artifact(str(encoded_data_save_path), mlflow_directory)

            reconstructed_data_save_path = Path(save_path, "reconstructed_data.csv")
            encoded_data.to_csv(reconstructed_data_save_path, index=False)
            mlflow.log_artifact(str(reconstructed_data_save_path), mlflow_directory)

        return encoded_data, reconstructed_data

    @staticmethod
    def encode_decode_ae_data(encoder, decoder, data: pd.DataFrame, markers: list, save_path: Path = None,
                              mlflow_directory: str = None, use_mlflow: bool = True) -> Tuple:
        """
        Encodes and decodes the remaining test dataset. Is then further used for evaluation of performance
        """

        if use_mlflow:
            if mlflow_directory is None or save_path is None:
                raise ValueError("When use_mlflow is true, save_path and mlflow_directory are mandatory")

        encoded = encoder.predict(data)
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
