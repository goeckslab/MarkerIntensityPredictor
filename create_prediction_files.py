import pandas as pd
from pathlib import Path
import os, shutil
from typing import Dict, List

MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']
biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
LGBM_PATHS = [
    Path("mesmer", "tumor_in_patient"),
    Path("mesmer", "tumor_exp_patient"),
]

AE_PATHS = [
    Path("ae_imputation", "ip", "zero"),
    Path("ae_imputation", "exp", "zero"),
]


def reset_predictions() -> Dict[str, List]:
    return {
        "9_2_1": [],
        "9_2_2": [],
        "9_3_1": [],
        "9_3_2": [],
        "9_14_1": [],
        "9_14_2": [],
        "9_15_1": [],
        "9_15_2": [],
    }


def create_lgbm_predictions():
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")
    columns.append("Experiment")

    predictions = pd.DataFrame(columns=columns)

    biopsy_predictions = reset_predictions()

    for folder_path in LGBM_PATHS:
        # iterate through all folders and subfolders
        print(folder_path)
        for root, dirs, _ in os.walk(folder_path):
            for directory in dirs:
                if directory not in biopsies:
                    continue

                for marker in MARKERS:
                    for _, experiment_runs, _ in os.walk(Path(root, directory, marker, "results")):
                        if "experiment_run" not in experiment_runs:
                            continue

                        for experiment_run in experiment_runs:
                            results_path = str(Path(root, directory, marker, "results", experiment_run)).split('/')
                            mode = "IP" if "in_patient" in results_path[-5] else "EXP"
                            biopsy = results_path[-4]
                            marker = results_path[-3]
                            experiment_id = 0 if results_path[-1] == "experiment_run" else int(
                                results_path[-1].split("_")[-1])

                            try:
                                marker_predictions = pd.read_csv(
                                    Path(root, directory, marker, "results", experiment_run,
                                         f"{marker}_predictions.csv"),
                                    header=None)
                            except:
                                continue

                            biopsy_dfs = biopsy_predictions[biopsy]
                            # check if there is already a dataframe for this experiment id
                            if len(biopsy_dfs) <= experiment_id:
                                print("adding new experiment df")
                                biopsy_dfs.append(pd.DataFrame(columns=columns))
                                # add marker data to dataframe
                                biopsy_dfs[experiment_id][marker] = marker_predictions[0]
                                # add biopsy id to dataframe
                                biopsy_dfs[experiment_id]["Biopsy"] = biopsy
                                # add mode to dataframe
                                biopsy_dfs[experiment_id]["Mode"] = mode
                                # add experiment id to dataframe
                                biopsy_dfs[experiment_id]["Experiment"] = experiment_id
                            else:
                                # add the marker data to the dataframe
                                biopsy_dfs[experiment_id][marker] = marker_predictions[0]

        # combine all biopsy dfs into one dataframe
        for biopsy in biopsy_predictions:
            for df in biopsy_predictions[biopsy]:
                predictions = pd.concat([predictions, df], ignore_index=True)

        biopsy_predictions = reset_predictions()
    predictions.to_csv(Path(save_path, "lgbm_predictions.csv"), index=False)


def create_en_predictions():
    pass


def create_ae_predictions():
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")
    columns.append("Experiment")

    predictions = pd.DataFrame(columns=columns)

    biopsy_predictions = reset_predictions()

    for folder_path in AE_PATHS:
        # print(folder_path)
        for root, dirs, _ in os.walk(folder_path):
            for directory in dirs:
                if directory not in biopsies and "no_noise" not in str(Path(root, directory)):
                    continue

                sub_path = Path(root, directory)
                for biopsy in biopsies:
                    for _, experiment_runs, _ in os.walk(Path(sub_path, biopsy, "no_hp", "0")):

                        for experiment_run in experiment_runs:
                            results_path = Path(sub_path, biopsy, "no_hp", "0", experiment_run)
                            results_path_splits = str(results_path).split('/')
                            mode = "IP" if "ip" in results_path_splits[-5] else "EXP"
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



                            except:
                                continue

                            biopsy_dfs = biopsy_predictions[biopsy]
                            # check if there is already a dataframe for this experiment id
                            print("adding new experiment df")
                            biopsy_dfs.append(pd.DataFrame(columns=columns, data=marker_predictions))
                            # add biopsy id to dataframe
                            biopsy_dfs[experiment_id]["Biopsy"] = biopsy
                            # add mode to dataframe
                            biopsy_dfs[experiment_id]["Mode"] = mode
                            # add experiment id to dataframe
                            biopsy_dfs[experiment_id]["Experiment"] = experiment_id

        # combine all biopsy dfs into one dataframe
        for biopsy in biopsy_predictions:
            for df in biopsy_predictions[biopsy]:
                predictions = pd.concat([predictions, df], ignore_index=True)

    predictions.to_csv(Path(save_path, "ae_predictions.csv"), index=False)


if __name__ == '__main__':

    save_path = Path("data/cleaned_data/predictions")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    create_ae_predictions()
    create_lgbm_predictions()
