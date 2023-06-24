import pandas as pd
from pathlib import Path
import os, shutil, argparse
from typing import Dict, List

MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
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


def create_lgbm_predictions():
    print("Creating LGBM predictions...")
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    predictions = pd.DataFrame(columns=columns)

    for folder_path in LGBM_PATHS:
        # iterate through all folders and subfolders
        mode = "IP" if "in_patient" in str(folder_path) else "EXP"
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

            experiment_biopsy_predictions = {}
            for marker in MARKERS:
                marker_dir = Path(biopsy_path, marker)

                for experiment_run in os.listdir(Path(marker_dir, "results")):
                    experiment_dir = Path(marker_dir, "results", experiment_run)
                    if not experiment_dir.is_dir():
                        continue

                    experiment_id = 0 if str(experiment_run).split('_')[-1] == "run" else str(experiment_run).split(
                        '_')[-1]

                    try:
                        marker_predictions = pd.read_csv(
                            Path(experiment_dir, f"predictions.csv"))

                    except BaseException as ex:
                        print(ex)
                        continue

                    if experiment_id in experiment_biopsy_predictions:
                        experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[
                            'prediction'].values
                        experiment_biopsy_predictions[experiment_id]["Experiment"] = experiment_id
                        experiment_biopsy_predictions[experiment_id]["Biopsy"] = biopsy
                        experiment_biopsy_predictions[experiment_id]["Mode"] = mode
                    else:
                        # Add new experiment id to dictionary
                        experiment_biopsy_predictions[experiment_id] = pd.DataFrame(columns=MARKERS)
                        # add marker data to dataframe
                        experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[
                            'prediction'].values
                        # add biopsy id to dataframe
                        experiment_biopsy_predictions[experiment_id]["Biopsy"] = biopsy
                        # add mode to dataframe
                        experiment_biopsy_predictions[experiment_id]["Mode"] = mode
                        # add experiment id to dataframe
                        experiment_biopsy_predictions[experiment_id]["Experiment"] = experiment_id

            # calculate mean prediction dataframe from all experiments for this biopsy
            biopsy_mean_predictions = pd.DataFrame(columns=MARKERS)
            for experiment in experiment_biopsy_predictions.keys():
                if len(biopsy_mean_predictions) == 0:
                    biopsy_mean_predictions = \
                        experiment_biopsy_predictions[experiment][MARKERS]

                else:
                    biopsy_mean_predictions = biopsy_mean_predictions + \
                                              experiment_biopsy_predictions[experiment][
                                                  MARKERS]

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
    predictions.to_csv(Path(save_path, "lgbm_predictions.csv"), index=False)


def create_en_predictions():
    all_mean_predictions = pd.DataFrame()
    predictions = pd.DataFrame()
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

            print(f"Loading biosy: {biopsy}...")
            experiment_biopsy_predictions = {}
            for marker in MARKERS:
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

                        if experiment_id in experiment_biopsy_predictions:
                            experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[0].values
                            experiment_biopsy_predictions[experiment_id]["Experiment"] = experiment_id
                            experiment_biopsy_predictions[experiment_id]["Biopsy"] = biopsy
                            experiment_biopsy_predictions[experiment_id]["Mode"] = mode
                        else:
                            # Add new experiment id to dictionary
                            experiment_biopsy_predictions[experiment_id] = pd.DataFrame(columns=MARKERS)
                            # add marker data to dataframe
                            experiment_biopsy_predictions[experiment_id][marker] = marker_predictions[0].values
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
            biopsy_mean_predictions = pd.DataFrame(columns=MARKERS)
            for experiment in experiment_biopsy_predictions.keys():
                if len(biopsy_mean_predictions) == 0:
                    biopsy_mean_predictions = \
                        experiment_biopsy_predictions[experiment][MARKERS]

                else:
                    biopsy_mean_predictions = biopsy_mean_predictions + experiment_biopsy_predictions[experiment][
                        MARKERS]

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
    predictions.to_csv(Path(save_path, "en_predictions.csv"), index=False)


def create_ae_predictions():
    print("Creating AE predictions...")
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    predictions = pd.DataFrame(columns=columns)

    for folder_path in AE_PATHS:
        mode = "IP" if "ip" in str(folder_path) else "EXP"

        # print(folder_path)
        for root, dirs, _ in os.walk(folder_path):
            for directory in dirs:
                if directory not in biopsies and "no_noise" not in str(Path(root, directory)):
                    continue

                sub_path = Path(root, directory)
                for biopsy in biopsies:
                    biopsy_predictions = {}
                    biopsy_path = Path(sub_path, biopsy, "no_hp", "0")

                    if not Path(biopsy_path).exists():
                        continue

                    for experiment_run in os.listdir(biopsy_path):
                        if not Path(biopsy_path, experiment_run).is_dir():
                            continue

                        results_path = Path(biopsy_path, experiment_run)
                        results_path_splits = str(results_path).split('/')
                        experiment_id = 0 if results_path_splits[-1] == "experiment_run" else int(
                            results_path_splits[-1].split("_")[-1])

                        try:
                            # Load prediction 5 - 9
                            marker_predictions_5 = pd.read_csv(
                                Path(results_path, "5_predictions.csv"))
                            marker_predictions_6 = pd.read_csv(
                                Path(results_path, "6_predictions.csv"))
                            marker_predictions_7 = pd.read_csv(
                                Path(results_path, "7_predictions.csv"))
                            marker_predictions_8 = pd.read_csv(
                                Path(results_path, "8_predictions.csv"))
                            marker_predictions_9 = pd.read_csv(
                                Path(results_path, "9_predictions.csv"))

                            # calculate mean per cell over all marker predictions
                            marker_predictions = (
                                                         marker_predictions_5 + marker_predictions_6 + marker_predictions_7 + marker_predictions_8 + marker_predictions_9) / 5


                        except BaseException as ex:
                            print("Could not load predictions")
                            print(ex)
                            continue

                        biopsy_predictions[experiment_id] = pd.DataFrame(columns=columns,
                                                                         data=marker_predictions)

                        # calculate mean prediction dataframe from all experiments for this biopsy
                    biopsy_mean_predictions = pd.DataFrame(columns=columns)
                    for experiment in biopsy_predictions.keys():
                        if len(biopsy_mean_predictions) == 0:
                            biopsy_mean_predictions = \
                                biopsy_predictions[experiment][columns]

                        else:
                            biopsy_mean_predictions = biopsy_mean_predictions + biopsy_predictions[experiment][
                                columns]

                    # divide by number of experiments
                    biopsy_mean_predictions = biopsy_mean_predictions / len(
                        biopsy_predictions.keys())

                    # add biopsy id to dataframe
                    biopsy_mean_predictions["Biopsy"] = biopsy
                    # add mode to dataframe
                    biopsy_mean_predictions["Mode"] = mode

                    # add experiment id to dataframe
                    predictions = pd.concat([predictions, biopsy_mean_predictions], ignore_index=True)

    predictions.to_csv(Path(save_path, "ae_predictions.csv"), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic_net", "-en", action="store_true", default=False)

    args = parser.parse_args()

    save_path = Path("data/cleaned_data/predictions")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if args.elastic_net:
        print("Creating elastic net predictions...")
        create_en_predictions()
    else:
        print("Creating lgbm & ae predictions...")
        create_lgbm_predictions()
        create_ae_predictions()
