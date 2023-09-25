import random
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
import os, argparse, logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Dict
from tqdm import tqdm

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

SHARED_SPATIAL_FEATURES = ['pRB_mean', "CD45_mean", "CK19_mean", "Ki67_mean", "aSMA_mean", "Ecad_mean", "PR_mean",
                           "CK14_mean", "HER2_mean", "AR_mean", "CK17_mean", "p21_mean", "Vimentin_mean", "pERK_mean",
                           "EGFR_mean", "ER_mean"]

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ae_imputation_m/debug.log"),
                        logging.StreamHandler()
                    ])

test_data_save_folder: Path = Path("data", "cleaned_data", "tma", "scaled_biopsies")


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "debug.log")
    file_logger = logging.FileHandler(save_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for handler in log.handlers[:]:  # remove all old handlers
        log.removeHandler(handler)
    log.addHandler(file_logger)
    log.addHandler(logging.StreamHandler())


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


def impute_markers(scores: List, test_data: pd.DataFrame, all_predictions: Dict, biopsy_name: str,
                   mode: str, spatial_radius: int, experiment_id: int, save_folder: str, file_name: str,
                   replace_value: str, iterations: int, store_predictions: bool, subset: int):
    try:
        for marker in SHARED_MARKERS:
            print(f"Imputing marker {marker}")
            # copy the test data
            input_data = test_data.copy()
            if replace_value == "zero":
                input_data[marker] = 0
                if int(spatial_radius) > 0:
                    input_data[f"{marker}_mean"] = 0
            elif replace_value == "mean":
                mean = input_data[marker].mean()
                input_data[marker] = mean
                if int(spatial_radius) > 0:
                    input_data[f"{marker}_mean"] = mean

            marker_prediction = input_data.copy()
            for iteration in range(iterations):

                predicted_intensities = ae.decoder.predict(ae.encoder.predict(marker_prediction))

                predicted_intensities = pd.DataFrame(data=predicted_intensities, columns=test_data.columns)
                if store_predictions:
                    all_predictions[iteration][marker] = predicted_intensities[marker].values

                imputed_marker = predicted_intensities[marker].values
                marker_prediction[marker] = imputed_marker

                scores.append({
                    "Marker": marker,
                    "Biopsy": biopsy_name,
                    "MAE": mean_absolute_error(marker_prediction[marker], test_data[marker]),
                    "RMSE": mean_squared_error(marker_prediction[marker], test_data[marker], squared=False),
                    "HP": 0,
                    "Mode": mode,
                    "Imputation": 1,
                    "Iteration": iteration,
                    "FE": spatial_radius,
                    "Experiment": int(f"{experiment_id}{subset}"),
                    "Network": "AE",
                    "Noise": 0,
                    "Replace Value": replace_value
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
        logging.error(ex)
        logging.error("Test truth:")
        logging.error(test_data)
        logging.error("Predictions:")
        logging.error(predictions)
        logging.error(mode)
        logging.error(spatial_radius)
        logging.error(experiment_id)
        logging.error(replace_value)

        raise


def create_results_folder(biopsy_name: str, spatial_radius: str) -> [Path, int]:
    save_folder = Path(f"ae_imputation_tma", "exp")

    save_folder = Path(save_folder, replace_value)

    save_folder = Path(save_folder, biopsy_name)
    save_folder = Path(save_folder, spatial_radius)

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
    if not test_data_save_folder.exists():
        test_data_save_folder.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", type=str, required=True,
                        help="Provide the biopsy folder path")
    parser.add_argument("-sp", "--spatial", required=False, default="0", choices=["0", "23", "46", "92", "138", "184"],
                        type=str)
    parser.add_argument("-o", "--override", action='store_true', default=False, help="Override existing hyperopt")
    parser.add_argument("-i", "--iterations", action="store", default=10, type=int)
    parser.add_argument("-rm", "--replace_mode", action="store", type=str, choices=["mean", "zero"], default="zero")
    parser.add_argument("-s", "--subsets", action="store", default=1, type=int, help="Number of subsets to use")
    args = parser.parse_args()

    iterations: int = args.iterations
    replace_value: str = args.replace_mode
    spatial: str = args.spatial
    subsets: int = args.subsets

    # Load test data
    test_biopsy_name: str = args.biopsy
    patient: str = "_".join(Path(test_biopsy_name).stem.split("_")[:2])

    scores_file_name = "scores.csv"

    save_folder, experiment_id = create_results_folder(biopsy_name=test_biopsy_name, spatial_radius=spatial)

    setup_log_file(save_path=save_folder)
    logging.debug("Replace value: " + replace_value)
    logging.debug(scores_file_name)

    # Load noisy train data
    train_data = []
    base_path = Path("data", "cleaned_data", "tma", "biopsies")
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        if file.endswith(".tsv") and patient not in file_name:
            logging.debug("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file), sep="\t")
            data = clean_column_names(data)
            train_data.append(data)

    train_data = pd.concat(train_data)

    if spatial != "0":
        logging.debug("Selecting marker and spatial information")
        # select shared markers as well as spatial shared features from the train data
        train_data = train_data[SHARED_MARKERS + SHARED_SPATIAL_FEATURES].copy()
    else:
        logging.debug("Selecting marker")
        train_data = train_data[SHARED_MARKERS].copy()

    # Load test data
    test_data = pd.read_csv(Path(base_path, f"{test_biopsy_name}.tsv"), sep="\t")
    test_data = clean_column_names(test_data)

    if spatial != "0":
        test_data = test_data[SHARED_MARKERS + SHARED_SPATIAL_FEATURES].copy()
        assert test_data.shape[1] == 32, "Test data not complete"
    else:
        test_data = test_data[SHARED_MARKERS].copy()
        assert test_data.shape[1] == 16, "Test data not complete"

    # Scale data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(train_data + 1)),
                              columns=train_data.columns)
    test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(test_data + 1)),
                             columns=test_data.columns)

    # save preprocessed test data
    test_data.to_csv(Path(test_data_save_folder, f"{test_biopsy_name}.tsv"), sep="\t", index=False)

    # Split noisy train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    scores = []
    predictions = {}
    for i in range(iterations):
        predictions[i] = pd.DataFrame(columns=SHARED_MARKERS)

    # Create ae
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    ae = AutoEncoder(input_dim=train_data.shape[1], latent_dim=5)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    history = ae.fit(train_data, train_data, epochs=100, batch_size=32, shuffle=True,
                     validation_data=(val_data, val_data), callbacks=callbacks)

    for i in tqdm(range(1, subsets)):
        # sample new dataset from test_data
        test_data_sample = test_data.sample(frac=0.7, random_state=random.randint(0, 100000), replace=True)

        # Predict
        impute_markers(scores=scores, all_predictions=predictions, mode="exp", spatial_radius=int(spatial),
                       experiment_id=experiment_id, replace_value=replace_value,
                       iterations=iterations,
                       store_predictions=False, test_data=test_data_sample, subset=i, file_name=scores_file_name,
                       save_folder=save_folder, biopsy_name=test_biopsy_name)

    predictions = impute_markers(scores=scores, all_predictions=predictions, mode="exp",
                                 spatial_radius=int(spatial),
                                 experiment_id=experiment_id, replace_value=replace_value,
                                 iterations=iterations,
                                 store_predictions=True, test_data=test_data, subset=0, file_name=scores_file_name,
                                 save_folder=save_folder, biopsy_name=test_biopsy_name)

    # Save results
    for key, value in predictions.items():
        value.to_csv(f"{save_folder}/{key}_predictions.csv",
                     index=False)
