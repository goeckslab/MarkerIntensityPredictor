import pandas as pd
from pathlib import Path
import os, shutil, argparse
from typing import Dict, List
import numpy as np

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
    lgbm_save_path = Path(save_path, "lgbm")
    if not lgbm_save_path.exists():
        lgbm_save_path.mkdir(parents=True)

    print("Creating LGBM predictions...")
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    predictions = pd.DataFrame(columns=columns)
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode", "Experiment"])

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

            # load ground truth data for biopsy
            print(f"Loading ground truth data for biopsy {biopsy}")
            ground_truth = pd.read_csv(
                Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep='\t')

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

                    # calculate mae for all quartiles
                    mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                        ground_truth=ground_truth, marker=marker, predictions=marker_predictions, std=2)

                    quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
                        {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                         "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                       quartiles[marker][0.75]], "Marker": marker,
                         "Biopsy": biopsy,
                         "Mode": mode,
                         "Load Path": str(experiment_dir),
                         "Experiment": experiment_id,
                         "Std": 2
                         })])

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
    predictions.to_csv(Path(lgbm_save_path, "predictions.csv"), index=False)

    # remove all rows with NaN values
    quartile_performance = quartile_performance.dropna()
    quartile_performance.to_csv(Path(lgbm_save_path, "quartile_performance.csv"), index=False)


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
                            experiment_biopsy_predictions[experiment_id] = pd.DataFrame(columns=MARKERS)
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
    predictions.to_csv(Path(en_save_path, "predictions.csv"), index=False)

    # remove all nan from quartile performance
    quartile_performance = quartile_performance.dropna()
    quartile_performance.to_csv(Path(en_save_path, "quartile_performance.csv"), index=False)


def create_ae_predictions(save_path: Path):
    ae_save_path = Path(save_path, "ae")
    if not ae_save_path.exists():
        ae_save_path.mkdir(parents=True)

    print("Creating AE predictions...")
    columns = [marker for marker in MARKERS]
    columns.append("Biopsy")
    columns.append("Mode")

    predictions = pd.DataFrame(columns=columns)
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode", "Experiment"])

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

                    print(f"Loading ground truth data for biopsy {biopsy}...")
                    ground_truth = pd.read_csv(
                        Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"),
                        sep='\t')
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

                            for marker in MARKERS:
                                # calculate mae for all quartiles
                                mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                                    ground_truth=ground_truth, marker=marker, predictions=marker_predictions, std=2)

                                quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
                                    {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                                     "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5],
                                                   quartiles[marker][0.75],
                                                   quartiles[marker][0.75]], "Marker": marker,
                                     "Biopsy": biopsy,
                                     "Mode": mode,
                                     "Load Path": str(results_path),
                                     "Experiment": experiment_id,
                                     "Std": 2
                                     })])

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

    predictions.to_csv(Path(ae_save_path, "predictions.csv"), index=False)

    # remove all nan from quartile performance
    quartile_performance = quartile_performance.dropna()
    quartile_performance.to_csv(Path(save_path, "quartile_performance.csv"), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--elastic_net", "-en", action="store_true", default=False)

    args = parser.parse_args()

    save_path = Path("data/cleaned_data/predictions")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if args.elastic_net:
        print("Creating elastic net predictions...")
        create_en_predictions(save_path=save_path)
    else:
        print("Creating lgbm & ae predictions...")
        create_lgbm_predictions(save_path=save_path)
        create_ae_predictions(save_path=save_path)
