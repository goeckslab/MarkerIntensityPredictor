import pandas as pd
from pathlib import Path
import os, shutil, argparse, logging, sys
from typing import Dict, List
import numpy as np
import datetime
import traceback
from ludwig.api import LudwigModel

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
LGBM_PATHS = [
    Path("mesmer", "tumor_in_patient"),
    Path("mesmer", "tumor_exp_patient"),
]

AE_PATHS = [
    Path("ae_imputation", "ip", "mean"),
    Path("ae_imputation", "exp", "mean"),
]

EN_PATHS = [
    Path("mesmer", "tumor_in_patient_en"),
    Path("mesmer", "tumor_exp_patient_en"),
]

AE_M_PATHS = [
    Path("ae_imputation_m", "ip", "mean"),
    Path("ae_imputation_m", "exp", "mean"),
]

VAE_ALL_PATHS = [
    Path("vae_imputation_all", "ip", "zero"),
    Path("vae_imputation_all", "exp", "zero"),
]

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("predictions/debug.log"),
                        logging.StreamHandler()
                    ])


def load_test_data_set(mode: str, biopsy: str, hyper: int, spatial_radius: int) -> pd.DataFrame:
    if mode == "ip":
        # change last number of biopsy to 1 if it is 2
        if biopsy[-1] == "2":
            test_biopsy_name = biopsy[:-1] + "1"
        else:
            test_biopsy_name = biopsy[:-1] + "2"

        logging.debug(biopsy)
        logging.debug(test_biopsy_name)
        assert test_biopsy_name[-1] != biopsy[-1], "The bx should not be the same"
        if spatial_radius == 0:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')

        else:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", f"tumor_mesmer_sp_{spatial_radius}", "preprocessed",
                     f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')


    else:
        test_biopsy_name = biopsy
        assert test_biopsy_name == biopsy, "The bx should be the same"
        logging.debug(test_biopsy_name)

        if spatial_radius == 0:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')

        else:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", f"tumor_mesmer_sp_{spatial_radius}", "preprocessed",
                     f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')

    return test_dataset


def calculate_quartile_performance(ground_truth: pd.DataFrame, marker: str, predictions: pd.DataFrame, std: int):
    if std > 0:
        # keep only the rows that are within 3 standard deviations of the mean
        ground_truth = ground_truth[
            np.abs(ground_truth[marker] - ground_truth[marker].mean()) <= (std * ground_truth[marker].std())].copy()

    # select all indexes of predictions which are in the ground truth index
    predictions = predictions.loc[ground_truth.index].copy()

    # extract the quartiles
    quartiles = ground_truth.quantile([0.25, 0.5, 0.75])

    # select the rows that are in the quartiles from the predictions and ground truth
    gt_quartile_1 = ground_truth[ground_truth[marker] <= quartiles[marker][0.25]]
    gt_quartile_2 = ground_truth[
        (ground_truth[marker] > quartiles[marker][0.25]) & (
                ground_truth[marker] <= quartiles[marker][0.5])]
    gt_quartile_3 = ground_truth[
        (ground_truth[marker] > quartiles[marker][0.5]) & (
                ground_truth[marker] <= quartiles[marker][0.75])]
    gt_quartile_4 = ground_truth[ground_truth[marker] > quartiles[marker][0.75]]

    pred_quartile_1 = predictions.loc[gt_quartile_1.index]
    pred_quartile_2 = predictions.loc[gt_quartile_2.index]
    pred_quartile_3 = predictions.loc[gt_quartile_3.index]
    pred_quartile_4 = predictions.loc[gt_quartile_4.index]

    # Calculate MAE for all quartiles
    mae_1 = np.mean(np.abs(gt_quartile_1[marker] - pred_quartile_1["prediction"]))
    mae_2 = np.mean(np.abs(gt_quartile_2[marker] - pred_quartile_2["prediction"]))
    mae_3 = np.mean(np.abs(gt_quartile_3[marker] - pred_quartile_3["prediction"]))
    mae_4 = np.mean(np.abs(gt_quartile_4[marker] - pred_quartile_4["prediction"]))

    return mae_1, mae_2, mae_3, mae_4, quartiles


def create_lgbm_predictions(save_path: Path):
    print("Creating LGBM predictions...")
    columns = [marker for marker in SHARED_MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    biopsy_predictions = {}
    biopsy_counter = {}
    predictions = pd.DataFrame(columns=columns)
    for load_path in LGBM_PATHS:
        for root, sub_directories, files in os.walk(load_path):
            for sub_directory in sub_directories:
                current_path = Path(root, sub_directory)
                if 'experiment_run' not in str(current_path) or 'experiment_run' != current_path.parts[-1]:
                    continue

                logging.debug("Current path: " + str(current_path))

                path_splits: [] = current_path.parts

                biopsy: str = path_splits[2]
                mode: str = "ip" if "_in_" in path_splits[1] else "exp"
                experiment_id: int = 0 if path_splits[-1] == "experiment_run" else int(path_splits[-1].split("_")[-1])
                radius: int = 0 if "_sp" not in path_splits[1] else int(path_splits[1].split("_")[-1])
                hyper = 1 if "_hyper" in path_splits[1] else 0
                protein = path_splits[3]

                logging.debug(f"Biopsy: {biopsy}")
                logging.debug(f"Mode: {mode}")
                logging.debug(f"Experiment ID: {experiment_id}")
                logging.debug(f"Radius: {radius}")
                logging.debug(f"Hyper: {hyper}")
                logging.debug(f"Protein: {protein}")

                assert mode == "ip" or mode == "exp", f"Mode {mode} not in ['ip', 'exp']"
                assert biopsy in biopsies, f"Biopsy {biopsy} not in biopsies"
                assert radius in [0, 23, 46, 92, 138, 184], f"Radius {radius} not in [0, 23,46,92,138,184]"
                assert hyper in [0, 1], f"Hyper {hyper} not in [0,1]"
                assert protein in SHARED_MARKERS, f"Protein {protein} not in SHARED_MARKERS"

                unique_key = f"{biopsy}||{mode}||{radius}||{hyper}"

                models = None
                try:
                    model = LudwigModel.load(str(Path(current_path, 'model')))
                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    sys.exit(0)
                except BaseException as ex:
                    logging.error(ex)
                    continue

                try:
                    test_data: pd.DataFrame = load_test_data_set(biopsy=biopsy, mode=mode, spatial_radius=radius,
                                                                 hyper=hyper)
                    # predict on test_data
                    protein_predictions, _ = model.predict(dataset=test_data)


                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    sys.exit(0)

                except BaseException as ex:
                    logging.error(f"Error occurred for experiment: {experiment_id}")
                    logging.error(f"Model loaded using path: {str(Path(current_path, 'model'))}")
                    logging.error(ex)
                    logging.error("Continuing to next experiment")
                    continue

                try:
                    if unique_key not in biopsy_predictions:
                        biopsy_counter[unique_key] = 1
                        biopsy_predictions[unique_key] = pd.DataFrame(columns=SHARED_MARKERS)
                        print("Key not in dict")
                        print(biopsy_predictions[unique_key])
                        input()

                        # add protein predictions to biopsy_predictions

                        print(protein_predictions[f"{protein}_predictions"].values)
                        input()
                        biopsy_predictions[unique_key][protein] = protein_predictions[f"{protein}_predictions"].values

                        print(biopsy_predictions[unique_key])
                        input()

                    else:
                        print("Key in dict")
                        biopsy_counter[unique_key] += 1
                        biopsy_temp_df = biopsy_predictions[unique_key].copy()
                        print(biopsy_predictions[unique_key])
                        input()

                        biopsy_predictions[unique_key][protein] = biopsy_temp_df[protein] + protein_predictions[
                            f"{protein}_predictions"].values
                        biopsy_predictions[unique_key][protein] = biopsy_predictions[unique_key][protein] / \
                                                                  biopsy_counter[unique_key]

                        print(biopsy_predictions[unique_key])
                        input()



                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    exit()

                except BaseException as ex:
                    logging.error(f"Error occurred at {datetime.datetime.now()}")
                    logging.error(f"Error saving predictions for protein {protein}")
                    logging.error(unique_key)
                    logging.error(ex)
                    logging.error(traceback.format_exc())
                    continue

    save_path = Path(save_path, "lgbm")

    for key in biopsy_predictions.keys():
        mean_biopsy_predictions = biopsy_predictions[key]
        # f"{biopsy}||{mode}||{radius}||{hyper}"
        key_splits = key.split("||")

        print(key_splits)
        biopsy = key_splits[0]
        mode = key_splits[1]
        fe = int(key_splits[2])
        hp = int(key_splits[3])

        mean_biopsy_predictions["Biopsy"] = biopsy
        mean_biopsy_predictions["Mode"] = mode
        mean_biopsy_predictions["FE"] = fe
        mean_biopsy_predictions["HP"] = hp

        # convert fe , hp and noise to int
        mean_biopsy_predictions["FE"] = mean_biopsy_predictions["FE"].astype(int)
        mean_biopsy_predictions["HP"] = mean_biopsy_predictions["HP"].astype(int)

        predictions = pd.concat([predictions, mean_biopsy_predictions], ignore_index=True)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    predictions["FE"] = predictions["FE"].astype(int)
    predictions["HP"] = predictions["HP"].astype(int)
    predictions.to_csv(Path(save_path, f"predictions.csv"), index=False)


def create_en_predictions(save_path: Path):
    en_save_path = Path(save_path, "en")
    if not en_save_path.exists():
        en_save_path.mkdir(parents=True)

    predictions = pd.DataFrame()
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode", "Experiment"])
    for folder_path in EN_PATHS:
        mode = "IP" if "in_patient" in str(folder_path) else "EXP"
        # print(folder_path)
        for directory in os.listdir(folder_path):
            if not Path(folder_path, directory).is_dir():
                continue
            biopsy_path = Path(folder_path, directory)
            if mode == "EXP":
                biopsy = str(biopsy_path).split('/')[-1]
            else:
                biopsy = str(biopsy_path).split('/')[-1]
                if biopsy[-1] == '1':
                    biopsy = biopsy[:-1] + '2'
                else:
                    biopsy = biopsy[:-1] + '1'

            print(f"Loading biopsy: {biopsy}...")
            ground_truth = pd.read_csv(
                Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"),
                sep='\t')

            experiment_biopsy_predictions = {}
            for marker in SHARED_MARKERS:
                print(f"Loading marker {marker} for biopsy {biopsy}...")
                marker_dir = Path(biopsy_path, biopsy, marker)
                for experiment_run in os.listdir(marker_dir):
                    if not Path(marker_dir, experiment_run).is_dir():
                        continue
                    experiment_run_dir = Path(marker_dir, experiment_run)
                    try:
                        experiment_id: int = int(str(experiment_run_dir).split('/')[-1].split('_')[-1])
                    except BaseException as ex:
                        print(ex)
                        print(experiment_run_dir)
                        print("Could not parse experiment id")
                        continue
                    try:
                        # load json file using json library
                        marker_predictions: pd.DataFrame = pd.read_csv(
                            Path(experiment_run_dir, f"{marker}_predictions.csv"),
                            header=None)

                        # rename column 0 to prediction
                        marker_predictions = marker_predictions.rename(columns={0: "prediction"})

                        # calculate mae for all quartiles
                        mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                            ground_truth=ground_truth, marker=marker, predictions=marker_predictions, std=2)

                        quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
                            {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                             "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                           quartiles[marker][0.75]], "Marker": marker,
                             "Biopsy": biopsy,
                             "Mode": mode,
                             "Load Path": str(experiment_run_dir),
                             "Experiment": experiment_id,
                             "Std": 2
                             })])

                        if experiment_id in experiment_biopsy_predictions:
                            experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[
                                "prediction"].values
                            experiment_biopsy_predictions[experiment_id]["Experiment"] = experiment_id
                            experiment_biopsy_predictions[experiment_id]["Biopsy"] = biopsy
                            experiment_biopsy_predictions[experiment_id]["Mode"] = mode
                        else:
                            # Add new experiment id to dictionary
                            experiment_biopsy_predictions[experiment_id] = pd.DataFrame(columns=SHARED_MARKERS)
                            # add marker data to dataframe
                            experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[
                                "prediction"].values
                            # add biopsy id to dataframe
                            experiment_biopsy_predictions[experiment_id]["Biopsy"] = biopsy
                            # add mode to dataframe
                            experiment_biopsy_predictions[experiment_id]["Mode"] = mode
                            # add experiment id to dataframe
                            experiment_biopsy_predictions[experiment_id]["Experiment"] = experiment_id



                    except KeyboardInterrupt:
                        exit()
                    except BaseException as ex:
                        print(ex)
                        print(experiment_run_dir)
                        print(experiment_id)
                        raise
            print(f"Merging predictions for biopsy {biopsy}...")

            # calculate mean prediction dataframe from all experiments for this biopsy
            biopsy_mean_predictions = pd.DataFrame(columns=SHARED_MARKERS)
            for experiment in experiment_biopsy_predictions.keys():
                if len(biopsy_mean_predictions) == 0:
                    biopsy_mean_predictions = \
                        experiment_biopsy_predictions[experiment][SHARED_MARKERS]

                else:
                    biopsy_mean_predictions = biopsy_mean_predictions + experiment_biopsy_predictions[experiment][
                        SHARED_MARKERS]

            # divide by number of experiments
            biopsy_mean_predictions = biopsy_mean_predictions / len(
                experiment_biopsy_predictions.keys())

            # add biopsy id to dataframe
            biopsy_mean_predictions["Biopsy"] = biopsy
            # add mode to dataframe
            biopsy_mean_predictions["Mode"] = mode

            predictions = pd.concat([predictions, biopsy_mean_predictions], ignore_index=True)

    print(predictions)
    print(predictions["Biopsy"].unique())
    print(predictions["Mode"].unique())
    # remove Experiment column from df
    predictions.to_csv(Path(en_save_path, "predictions.csv"), index=False)

    # remove all nan from quartile performance
    quartile_performance = quartile_performance.dropna()
    quartile_performance.to_csv(Path(en_save_path, "quartile_performance.csv"), index=False)


def create_ae_predictions(save_path: Path, imputation_mode: str = None):
    logging.info(f"Creating AE predictions for {imputation_mode}...")
    columns = [marker for marker in SHARED_MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    biopsy_predictions = {}
    biopsy_counter = {}
    predictions = pd.DataFrame(columns=columns)
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode", "Experiment"])

    if imputation_mode is None:
        load_paths = AE_PATHS
    elif imputation_mode == "Multi":
        load_paths = AE_M_PATHS
    else:
        load_paths = VAE_ALL_PATHS

    for load_path in load_paths:
        # print(folder_path)
        for root, sub_directories, files in os.walk(load_path):
            for sub_directory in sub_directories:
                current_path = Path(root, sub_directory)
                if 'experiment_run' not in str(current_path):  # or int(current_path.stem.split('_')[-1]) > 30:
                    continue

                logging.debug("Current path: " + str(current_path))

                try:
                    scores = pd.read_csv(Path(current_path, "scores.csv"))
                except FileNotFoundError:
                    logging.debug(f"Scores not found for {current_path}")
                    continue

                except KeyboardInterrupt:
                    logging.debug("Keyboard interrupt")
                    exit()

                except BaseException as ex:
                    logging.error(f"Error occurred at {datetime.datetime.now()}")
                    logging.error(f"Error loading score files for path {current_path}")
                    logging.error(ex)
                    logging.error(traceback.format_exc())
                    continue

                biopsy: str = scores["Biopsy"].values[0]
                mode: str = scores["Mode"].values[0]
                experiment_id: int = int(scores["Experiment"].values[0])
                replace_value: str = scores["Replace Value"].values[0]
                noise: int = int(scores["Noise"].values[0])
                radius: int = int(scores["FE"].values[0])
                hyper = int(scores["HP"].values[0])

                assert mode == "ip" or mode == "exp", f"Mode {mode} not in ['ip', 'exp']"
                assert biopsy in biopsies, f"Biopsy {biopsy} not in biopsies"
                assert radius in [0, 23, 46, 92, 138, 184], f"Noise {noise} not in [0, 23,46,92,138,184]"
                assert hyper in [0, 1], f"Hyper {hyper} not in [0,1]"
                assert replace_value in ["mean", "zero"], f"Replace value {replace_value} not in ['mean', 'zero']"

                try:
                    logging.debug(f"Loading ground truth data for biopsy {biopsy}...")
                    ground_truth = pd.read_csv(
                        Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"),
                        sep='\t')
                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    exit()

                except BaseException as ex:
                    logging.error(f"Error occurred at {datetime.datetime.now()}")
                    logging.error(f"Error loading ground truth data for biopsy {biopsy}")
                    logging.error(ex)
                    logging.error(traceback.format_exc())
                    continue

                try:
                    # Load prediction 5 - 9
                    marker_predictions_5 = pd.read_csv(
                        Path(current_path, "5_predictions.csv"))
                    marker_predictions_6 = pd.read_csv(
                        Path(current_path, "6_predictions.csv"))
                    marker_predictions_7 = pd.read_csv(
                        Path(current_path, "7_predictions.csv"))
                    marker_predictions_8 = pd.read_csv(
                        Path(current_path, "8_predictions.csv"))
                    marker_predictions_9 = pd.read_csv(
                        Path(current_path, "9_predictions.csv"))

                    # calculate mean per cell over all marker predictions
                    marker_predictions = (marker_predictions_5 + marker_predictions_6 + marker_predictions_7
                                          + marker_predictions_8 + marker_predictions_9) / 5

                    marker_predictions = marker_predictions[SHARED_MARKERS].copy()

                    unique_key = f"{biopsy}||{mode}||{replace_value}||{noise}||{radius}||{hyper}"

                    if unique_key not in biopsy_predictions:
                        biopsy_counter[unique_key] = 1
                        biopsy_predictions[unique_key] = marker_predictions

                    else:
                        biopsy_counter[unique_key] += 1
                        biopsy_temp_df = biopsy_predictions[unique_key]
                        biopsy_predictions[unique_key] = biopsy_temp_df + marker_predictions
                        biopsy_predictions[unique_key] = biopsy_predictions[unique_key] / biopsy_counter[unique_key]



                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    exit()

                except BaseException as ex:
                    logging.error(f"Error occurred at {datetime.datetime.now()}")
                    logging.error(f"Error loading prediction files for path {current_path}")
                    logging.error(ex)
                    logging.error(traceback.format_exc())
                    continue

    if imputation_mode is None:
        save_path = Path(save_path, "ae")
    elif imputation_mode == "Multi":
        save_path = Path(save_path, "ae_m")
    else:
        save_path = Path(save_path, "vae_all")
    for key in biopsy_predictions.keys():
        mean_biopsy_predictions = biopsy_predictions[key]
        # f"{biopsy}_{mode}_{replace_value}_{noise}_{radius}_{hyper}"
        key_splits = key.split("||")

        print(key_splits)
        biopsy = key_splits[0]
        mode = key_splits[1]
        replace_value = key_splits[2]
        noise = int(key_splits[3])
        fe = int(key_splits[4])
        hp = int(key_splits[5])

        mean_biopsy_predictions["Biopsy"] = biopsy
        mean_biopsy_predictions["Mode"] = mode
        mean_biopsy_predictions["Replace Value"] = replace_value
        mean_biopsy_predictions["Noise"] = noise
        mean_biopsy_predictions["FE"] = fe
        mean_biopsy_predictions["HP"] = hp

        # convert fe , hp and noise to int
        mean_biopsy_predictions["FE"] = mean_biopsy_predictions["FE"].astype(int)
        mean_biopsy_predictions["HP"] = mean_biopsy_predictions["HP"].astype(int)
        mean_biopsy_predictions["Noise"] = mean_biopsy_predictions["Noise"].astype(int)

        predictions = pd.concat([predictions, mean_biopsy_predictions], ignore_index=True)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(Path(save_path, f"predictions.csv"), index=False)


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "debug.log")

    if save_file.exists():
        save_file.unlink()

    file_logger = logging.FileHandler(save_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for handler in log.handlers[:]:  # remove all old handlers
        log.removeHandler(handler)
    log.addHandler(file_logger)
    log.addHandler(logging.StreamHandler())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-em", action="store", choices=["EN", "LGBM", "AE", "AE M", "VAE ALL"])
    args = parser.parse_args()

    model: str = args.model

    save_path = Path("data/cleaned_data/predictions")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    if model == "EN":
        print("Creating elastic net predictions...")
        create_en_predictions(save_path=save_path)
    elif model == "LGBM":
        create_lgbm_predictions(save_path=save_path)
    elif model == "AE":
        create_ae_predictions(save_path=save_path)
    elif model == "AE M":
        create_ae_predictions(save_path=save_path, imputation_mode="Multi")

    elif model == "VAE ALL":
        create_ae_predictions(save_path=save_path, imputation_mode="VAE ALL")
