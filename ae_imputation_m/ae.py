import random

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import shutil
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

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


def create_noise(shape: [], columns: List[str]):
    mu, sigma = 0, 0.1
    # creating a noise with the same dimension as the dataset (2,2)
    return pd.DataFrame(data=np.random.normal(mu, sigma, shape), columns=columns)


def save_scores(save_folder: str, file_name: str, new_scores: pd.DataFrame):
    if Path(f"{save_folder}/{file_name}").exists():
        temp_df = pd.read_csv(f"{save_folder}/{file_name}")
        new_scores = pd.concat([temp_df, pd.DataFrame(new_scores)])
    new_scores.to_csv(f"{save_folder}/{file_name}", index=False)


def impute_markers(scores: List, test_data: pd.DataFrame, all_predictions: Dict,
                   mode: str, experiment_id: int, save_folder: str, file_name: str,
                   replace_value: str, iterations: int, store_predictions: bool, subset: int, rounds: pd.DataFrame):
    try:
        for round in rounds["Round"].unique():
            markers = rounds[rounds["Round"] == round]["Marker"].values
            print(f"Imputing markers {markers}")
            # copy the test data
            input_data = test_data.copy()
            if replace_value == "zero":
                for marker in markers:
                    input_data[marker] = 0
            elif replace_value == "mean":
                for marker in markers:
                    input_data[marker] = input_data[marker].mean()

            marker_prediction = input_data.copy()
            for iteration in range(iterations):
                predicted_intensities = ae.decoder.predict(ae.encoder.predict(marker_prediction))

                predicted_intensities = pd.DataFrame(data=predicted_intensities, columns=test_data.columns)
                if store_predictions:
                    for marker in markers:
                        all_predictions[iteration][marker] = predicted_intensities[marker].values

                marker_prediction = input_data.copy()
                for marker in markers:
                    imputed_marker = predicted_intensities[marker].values
                    marker_prediction[marker] = imputed_marker

                for marker in markers:
                    scores.append({
                        "Marker": marker,
                        "Biopsy": test_biopsy_name,
                        "MAE": mean_absolute_error(marker_prediction[marker], test_data[marker]),
                        "RMSE": mean_squared_error(marker_prediction[marker], test_data[marker], squared=False),
                        "HP": 0,
                        "Mode": mode,
                        "Imputation": 1,
                        "Iteration": iteration,
                        "FE": 0,
                        "Experiment": int(f"{experiment_id}{subset}"),
                        "Network": "AE",
                        "Noise": 0,
                        "Replace Value": replace_value,
                        "Round": round
                    })

                if iteration % 20 == 0:
                    print("Performing temp save...")
                    save_scores(save_folder=save_folder, file_name=file_name, new_scores=pd.DataFrame(scores))
                    scores = []

        if len(scores) > 0:
            print("Saving remaining scores...")
            save_scores(save_folder=save_folder, file_name=file_name, new_scores=pd.DataFrame(scores))
            scores = []
        return all_predictions

    except KeyboardInterrupt as ex:
        print("Keyboard interrupt detected.")
        print("Saving scores...")
        if len(scores) > 0:
            save_scores(new_scores=pd.DataFrame(scores), save_folder=save_folder, file_name=file_name)
        raise

    except Exception as ex:
        print(ex)
        print("Test truth:")
        print(test_data)
        print("Predictions:")
        print(predictions)
        print(mode)
        print(experiment_id)
        print(replace_value)
        print(rounds)
        raise


def create_results_folder() -> [Path, int]:
    save_folder = Path(f"ae_imputation_m", patient_type)
    save_folder = Path(save_folder, replace_value)
    save_folder = Path(save_folder, test_biopsy_name)

    experiment_id = 0

    base_path = Path(save_folder, "experiment_run")
    save_path = Path(str(base_path) + "_" + str(experiment_id))
    while Path(save_path).exists():
        save_path = Path(str(base_path) + "_" + str(experiment_id))
        experiment_id += 1

    created: bool = False
    if not save_path.exists():
        while not created:
            try:
                save_path.mkdir(parents=True)
                created = True
            except:
                experiment_id += 1
                save_path = Path(str(base_path) + "_" + str(experiment_id))

    return save_path, experiment_id - 1 if experiment_id != 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", type=str, required=True,
                        help="Provide the biopsy name in the following format: 9_2_1. No suffix etc")
    parser.add_argument("-m", "--mode", choices=["ip", "exp"], default="ip")
    parser.add_argument("-i", "--iterations", action="store", default=10, type=int)
    parser.add_argument("-rm", "--replace_mode", action="store", type=str, choices=["mean", "zero"], default="zero")
    parser.add_argument("-r", "--repetitions", action="store", type=int, default=1)
    args = parser.parse_args()

    patient_type = args.mode
    iterations: int = args.iterations
    replace_value: str = args.replace_mode
    repetitions: int = args.repetitions

    print("Replace value: ", replace_value)

    # Load test data
    test_biopsy_name = args.biopsy
    patient: str = "_".join(Path(test_biopsy_name).stem.split("_")[:2])
    scores_file_name = "scores.csv"
    print(scores_file_name)

    save_folder, experiment_id = create_results_folder()

    rounds_per_marker = pd.read_csv(Path("data", "cleaned_data", "rounds", "rounds.csv"))

    # select only the rounds for the given patient
    rounds_per_marker = rounds_per_marker[rounds_per_marker["Patient"] == patient.replace("_", " ")].copy()
    # select only shared markers
    rounds_per_marker = rounds_per_marker[rounds_per_marker["Marker"].isin(SHARED_MARKERS)].copy()
    # reset index
    rounds_per_marker.reset_index(drop=True, inplace=True)

    # Load train data
    if patient_type == "ip":
        # Extract biopsy name
        test_biopsy_name_split = test_biopsy_name.split("_")
        if int(test_biopsy_name_split[2]) == 2:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_1"
        else:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_2"

        print(f"Mode: {patient_type}")
        print(f"Test biopsy being loaded: {test_biopsy_name}")
        print(f"Train biopsy being loaded: {train_biopsy_name}")

        base_path = "data/tumor_mesmer"
        # Load train data
        train_data = pd.read_csv(f'{base_path}/{train_biopsy_name}.csv')
        train_data = clean_column_names(train_data)

        print("Selecting marker")
        train_data = train_data[SHARED_MARKERS].copy()

        test_data = pd.read_csv(f'{base_path}/{test_biopsy_name}.csv')
        test_data = clean_column_names(test_data)

        test_data = test_data[SHARED_MARKERS].copy()
        assert test_data.shape[1] == 16, "Test data not complete"

    elif patient_type == "exp":
        # Load noisy train data
        train_data = []
        base_path = "data/tumor_mesmer"
        for file in os.listdir(base_path):
            file_name = Path(file).stem
            if file.endswith(".csv") and patient not in file_name:
                print("Loading train file: " + file)
                data = pd.read_csv(Path(base_path, file))
                data = clean_column_names(data)
                train_data.append(data)

        assert len(train_data) == 6, f"There should be 6 train datasets, loaded {len(train_data)}"
        train_data = pd.concat(train_data)

        print("Selecting marker")
        train_data = train_data[SHARED_MARKERS].copy()

        # Load test data
        test_data = pd.read_csv(Path(f'{base_path}/{test_biopsy_name}.csv'))
        test_data = clean_column_names(test_data)

        test_data = test_data[SHARED_MARKERS].copy()
        assert test_data.shape[1] == 16, "Test data not complete"



    else:
        raise ValueError(f"Unknown patient type: {patient_type}")

    # Scale data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(train_data + 1)),
                              columns=train_data.columns)
    test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(test_data + 1)),
                             columns=test_data.columns)

    # Split noisy train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    scores = []
    predictions = {}
    for i in range(iterations):
        predictions[i] = pd.DataFrame(columns=test_data.columns)

    # Create ae
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    ae = AutoEncoder(input_dim=train_data.shape[1], latent_dim=5)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    history = ae.fit(train_data, train_data, epochs=100, batch_size=32, shuffle=True,
                     validation_data=(val_data, val_data), callbacks=callbacks)

    for i in tqdm(range(1, repetitions)):
        # sample new dataset from test_data
        test_data_sample = test_data.sample(frac=0.7, random_state=random.randint(0, 100000), replace=True)

        # plot distribution of test_data_sample and test_data
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # sns.histplot(test_data_sample, ax=ax[0])
        # sns.histplot(test_data, ax=ax[1])
        # plt.show()

        # Predict
        impute_markers(scores=scores, all_predictions=predictions, mode=patient_type,
                       experiment_id=experiment_id, replace_value=replace_value,
                       iterations=iterations,
                       store_predictions=False, test_data=test_data_sample, subset=i, file_name=scores_file_name,
                       save_folder=save_folder, rounds=rounds_per_marker)

    predictions = impute_markers(scores=scores, all_predictions=predictions, mode=patient_type,

                                 experiment_id=experiment_id, replace_value=replace_value,
                                 iterations=iterations,
                                 store_predictions=True, test_data=test_data, subset=0, file_name=scores_file_name,
                                 save_folder=save_folder, rounds=rounds_per_marker)

    # Save results
    for key, value in predictions.items():
        value.to_csv(f"{save_folder}/{key}_predictions.csv",
                     index=False)
