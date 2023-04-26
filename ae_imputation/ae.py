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


class Hyperopt:
    @staticmethod
    def build_model(hp):
        input_layer = Input(shape=(16,))

        # Encoder
        encoded = Dense(units=hp.Int('enc_1', min_value=8, max_value=16, step=2), activation='relu')(input_layer)
        encoded = Dense(units=hp.Int('enc_2', min_value=5, max_value=8, step=1), activation='relu')(encoded)

        latent_space = Dense(units=hp.Int('latent_space', min_value=5, max_value=10, step=2))(encoded)

        # Decoder
        decoded = Dense(units=hp.Int('dec_1', min_value=5, max_value=8, step=1), activation='relu')(latent_space)
        decoded = Dense(units=hp.Int('dec_2', min_value=8, max_value=16, step=2), activation='relu')(decoded)
        decoded = Dense(units=16, activation='relu')(decoded)

        # Autoencoder
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        # Compile the model
        autoencoder.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=MeanSquaredError())

        return autoencoder

    @staticmethod
    def run(hyperopt_project_name: str, train_data, val_data):
        tuner = kt.Hyperband(Hyperopt.build_model,
                             objective="val_loss",
                             max_epochs=50,
                             factor=3,
                             directory=hyperopt_directory,
                             project_name=hyperopt_project_name,
                             hyperband_iterations=2)
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(train_data, train_data, epochs=50, validation_data=(val_data, val_data),
                     callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_data, train_data, epochs=50,
                            validation_data=(val_data, val_data), callbacks=[stop_early])

        return model


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
    parser.add_argument("-hp", "--hyper", action="store_true", required=False, default=False,
                        help="Should hyper parameter tuning be used?")
    parser.add_argument("-o", "--override", action='store_true', default=False, help="Override existing hyperopt")
    parser.add_argument("-i", "--iterations", action="store", default=5, type=int)
    args = parser.parse_args()

    mode = args.mode
    hp: bool = args.hyper
    iterations: int = args.iterations

    if hp:
        print("Using hyper parameter tuning")
        hyperopt_directory = Path("ae/ae_hyperopt")
        if not hyperopt_directory.exists():
            hyperopt_directory.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_biopsy_name = args.biopsy

    hyperopt_project_name = f"{test_biopsy_name}_{mode}_hp"
    if hp and args.override:
        print("Overriding hyperopt")
        shutil.rmtree(f"{hyperopt_directory}/{hyperopt_project_name}")

    # Load train data
    if mode == "ip":
        # Extract biopsy name
        test_biopsy_name_split = test_biopsy_name.split("_")
        if int(test_biopsy_name_split[2]) == 2:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_1"
        else:
            train_biopsy_name = "_".join(test_biopsy_name_split[:2]) + "_2"

        print("Test biopsy being loaded:: " + test_biopsy_name + " for mode: " + mode)
        print(f"Train biopsy being loaded: {train_biopsy_name}")

        # Load train data
        train_data = pd.read_csv(f'data/tumor_mesmer/{train_biopsy_name}.csv')
        train_data = clean_column_names(train_data)
        train_data = train_data[SHARED_MARKERS].copy()

        test_data = pd.read_csv(f'data/tumor_mesmer/{test_biopsy_name}.csv')
        test_data = clean_column_names(test_data)
        test_data = test_data[SHARED_MARKERS].copy()
    else:
        # Load noisy train data
        train_data = []
        search_dir = "data/tumor_mesmer"
        for file in os.listdir(search_dir):
            file_name = Path(file).stem
            if file.endswith(".csv") and file_name != test_biopsy_name:
                print("Loading train file: " + file)
                data = pd.read_csv(Path(search_dir, file))
                data = clean_column_names(data)
                train_data.append(data)

        assert len(train_data) == 7, f"There should be 7 train datasets, loaded {len(train_data)}"
        train_data = pd.concat(train_data)
        train_data = train_data[SHARED_MARKERS].copy()

        # Load test data
        test_data = pd.read_csv(f'data/tumor_mesmer/{test_biopsy_name}.csv')
        test_data = clean_column_names(test_data)
        test_data = test_data[SHARED_MARKERS].copy()

    # Scale data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(train_data + 1)),
                              columns=train_data.columns)
    test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(test_data + 1)),
                             columns=test_data.columns)

    # Split noisy train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    if not hp:
        # Create ae
        callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
        ae = AutoEncoder(input_dim=train_data.shape[1], latent_dim=5)
        ae.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        history = ae.fit(train_data, train_data, epochs=100, batch_size=32, shuffle=True,
                         validation_data=(val_data, val_data), callbacks=callbacks)

        predictions = pd.DataFrame(columns=test_data.columns)
        for marker in train_data.columns:
            # copy the test data
            input_data = test_data.copy()
            input_data[marker] = 0

            marker_prediction = input_data
            for i in range(iterations):
                marker_prediction = ae.decoder.predict(ae.encoder.predict(marker_prediction))

            # Add marker to prediction dataset

            marker_prediction = pd.DataFrame(data=marker_prediction, columns=test_data.columns)
            predictions[marker] = marker_prediction[marker].values

    else:
        # Hyperopt section
        ae = Hyperopt.run(hyperopt_project_name=hyperopt_project_name, train_data=train_data, val_data=val_data)

        predictions = pd.DataFrame(columns=test_data.columns)
        for marker in train_data.columns:
            # copy the test data
            input_data = test_data.copy()
            input_data[marker] = 0

            marker_prediction = input_data
            for i in range(iterations):
                marker_prediction = ae.predict(marker_prediction)

            # Add marker to prediction dataset
            predictions[marker] = marker_prediction[marker].values

    print(predictions)

    # Calculate mae for each marker between predictions and clean test data
    scores = []
    for marker in SHARED_MARKERS:
        scores.append({
            "Marker": marker,
            "Biopsy": test_biopsy_name,
            "MAE": mean_absolute_error(predictions[marker], test_data[marker]),
            "RMSE": mean_squared_error(predictions[marker], test_data[marker], squared=False),
            "HP": int(hp),
            "Mode": mode,
            "Imputation": 1,
            "Iterations": iterations
        })

    # Convert to df
    scores = pd.DataFrame(scores)

    save_folder = Path(f"ae_imputation/{mode}/{test_biopsy_name}")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Save results
    if not hp:
        scores.to_csv(f"{save_folder}/scores.csv", index=False)
        predictions.to_csv(f"{save_folder}/predictions.csv", index=False)
    else:
        scores.to_csv(f"{save_folder}/hp_scores.csv", index=False)
        predictions.to_csv(f"{save_folder}/hp_predictions.csv", index=False)
