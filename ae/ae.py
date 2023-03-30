from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
import os, argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


class AutoEncoder(Model):
    def __init__(self, input_dim: int, latent_dim: int):
        super(AutoEncoder, self).__init__()
        self._latent_dim = latent_dim
        self.encoder: Model = Sequential([
            Dense(input_dim, activation='relu', name="input"),
            Dense(self._latent_dim, name="latent_space"),
        ])
        self.decoder: Model = Sequential([
            Dense(input_dim, activation="relu", name="output"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", type=str, required=True,
                        help="Provide the biopsy name in the following format: 9_2_1. No suffix etc")
    parser.add_argument("-m", "--mode", required=True, choices=["ip", "op"])
    args = parser.parse_args()

    mode = args.mode

    # Load test data
    test_biopsy_name = args.biopsy
    noisy_test_data = pd.read_csv(f"data/tumor_mesmer_non_snr/{test_biopsy_name}.csv")
    noisy_test_data = clean_column_names(noisy_test_data)
    noisy_test_data = noisy_test_data[SHARED_MARKERS].copy()

    clean_test_data = pd.read_csv(f'data/tumor_mesmer/{test_biopsy_name}.csv')
    clean_test_data = clean_column_names(clean_test_data)
    clean_test_data = clean_test_data[SHARED_MARKERS].copy()

    # Load test data
    if mode == "ip":
        # Extract biopsy name
        test_biopsy_name_split = test_biopsy_name.split("_")

        if test_biopsy_name_split[2] == 2:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_1"
        else:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_2"

        print("Loading train data from biopsy: " + train_biopsy_name + " for mode: " + mode)

        # Load noisy train data
        noisy_train_data = pd.read_csv(f'data/tumor_mesmer_non_snr/{train_biopsy_name}.csv')
        noisy_train_data = clean_column_names(noisy_train_data)
        noisy_train_data = noisy_train_data[SHARED_MARKERS].copy()
        # Load clean train data
        clean_train_data = pd.read_csv(f'data/tumor_mesmer/{train_biopsy_name}.csv')
        clean_train_data = clean_column_names(clean_train_data)
        clean_train_data = clean_train_data[SHARED_MARKERS].copy()
    else:
        # Load noisy train data
        noisy_train_data = []
        for file in os.listdir("data/tumor_mesmer_non_snr"):
            if file.endswith(".csv") and file != test_biopsy_name:
                print("Loading noisy train file:" + file)
                noisy_train_data.append(pd.read_csv(pd.read_csv(args.folder / file)[SHARED_MARKERS]))

        assert len(noisy_train_data) == 7, "There should be 7 noisy train files"
        noisy_train_data = pd.concat(noisy_train_data)
        noisy_train_data = clean_column_names(noisy_train_data)
        noisy_train_data = noisy_train_data[SHARED_MARKERS].copy()

        # Load clean train data
        clean_train_data = []
        for file in os.listdir("data/tumor_mesmer"):
            if file.endswith(".csv") and file != test_biopsy_name:
                print("Loading clean train file:" + file)
                clean_train_data.append(pd.read_csv(pd.read_csv(args.folder / file)[SHARED_MARKERS]))

        assert len(clean_train_data) == 7, "There should be 7 clean train files"
        clean_train_data = pd.concat(clean_train_data)
        clean_train_data = clean_column_names(clean_train_data)
        clean_train_data = clean_train_data[SHARED_MARKERS].copy()

    # Scale data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    noisy_train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(noisy_train_data + 1)),
                                    columns=noisy_train_data.columns)
    noisy_test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(noisy_test_data + 1)),
                                   columns=noisy_test_data.columns)
    clean_train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(clean_train_data + 1)),
                                    columns=clean_train_data.columns)
    clean_test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(clean_test_data + 1)),
                                   columns=clean_test_data.columns)

    # Split noisy train data into train and validation
    noisy_train_data, noisy_val_data = train_test_split(noisy_train_data, test_size=0.2, random_state=42)
    clean_train_data, clean_val_data = train_test_split(clean_train_data, test_size=0.2, random_state=42)

    # Create ae
    callbacks = [EarlyStopping(monitor='loss', patience=5)]
    ae = AutoEncoder(input_dim=noisy_train_data.shape[1], latent_dim=5)
    ae.compile(optimizer="adam", loss=MeanSquaredError())
    history = ae.fit(noisy_train_data, clean_train_data, epochs=100, batch_size=32, shuffle=True,
                     validation_data=(noisy_val_data, clean_val_data), callbacks=callbacks)

    # Predict on test data
    predictions: pd.DataFrame = pd.DataFrame(data=ae.decoder.predict(ae.encoder.predict(noisy_test_data)),
                                             columns=noisy_test_data.columns)

    print(predictions)
    print(clean_test_data)

    # Calculate mae for each marker between predictions and clean test data

    mae_scores = []
    rmse_scores = []
    for marker in SHARED_MARKERS:
        mae_scores.append({
            "Marker": marker,
            "MAE": mean_absolute_error(predictions[marker], clean_test_data[marker]),
        })
        rmse_scores.append({
            "Marker": marker,
            "RMSE": mean_squared_error(predictions[marker], clean_test_data[marker], squared=False),
        })

    # Convert to df
    mae_scores = pd.DataFrame(mae_scores)
    rmse_scores = pd.DataFrame(rmse_scores)

    save_folder = Path(f"ae/{mode}/{test_biopsy_name}")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Save results
    mae_scores.to_csv(f"{save_folder}/mae_scores.csv", index=False)
    rmse_scores.to_csv(f"{save_folder}/rmse_scores.csv", index=False)
