import os, sys, logging, argparse
from pathlib import Path
import pandas as pd
import numpy as np

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("quartile_performance/debug.log"),
                        logging.StreamHandler()
                    ])


def load_test_data_set(mode: str, biopsy: str, spatial_radius: int) -> pd.DataFrame:
    if mode == "ip":
        # change last number of biopsy to 1 if it is 2
        if biopsy[-1] == "2":
            test_biopsy_name = biopsy[:-1] + "1"
        else:
            test_biopsy_name = biopsy[:-1] + "2"

        logging.debug(f"Spatial: {spatial_radius}")
        logging.debug(f"Mode {mode}")
        logging.debug(f"Biopsy {biopsy}")
        logging.debug(f"Test Biopsy Name: {test_biopsy_name}")
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
    mae_1 = np.mean(np.abs(gt_quartile_1[marker] - pred_quartile_1[marker]))
    mae_2 = np.mean(np.abs(gt_quartile_2[marker] - pred_quartile_2[marker]))
    mae_3 = np.mean(np.abs(gt_quartile_3[marker] - pred_quartile_3[marker]))
    mae_4 = np.mean(np.abs(gt_quartile_4[marker] - pred_quartile_4[marker]))

    return mae_1, mae_2, mae_3, mae_4, quartiles


def create_lgbm_quartile_performance(save_path: Path):
    predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "lgbm", "predictions.csv"))
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode", "Experiment"])

    # convert mode to upper cose in predictions
    predictions["Mode"] = predictions["Mode"].str.upper()

    for biopsy in predictions["Biopsy"].unique():
        for mode in predictions["Mode"].unique():
            biopsy_predictions = predictions[predictions["Biopsy"] == biopsy].copy()
            # select only exp
            biopsy_predictions = biopsy_predictions[biopsy_predictions["Mode"] == mode].copy()
            # select only FE  == 0
            if "FE" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["FE"] == 0].copy()
            # select no hyper
            if "HP" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["HP"] == 0].copy()

            if "Noise" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["Noise"] == 0].copy()

            biopsy_predictions.reset_index(drop=True, inplace=True)
            ground_truth: pd.DataFrame = load_test_data_set("exp", biopsy, 0)

            # remove biopsy mode FE and HP columns from biopsy predictions
            if "FE" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["FE"], inplace=True)
            if "HP" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["HP"], inplace=True)
            if "Noise" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Noise"], inplace=True)
            if "Mode" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Mode"], inplace=True)
            if "Biopsy" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Biopsy"], inplace=True)

            print(ground_truth)
            input()

            print(biopsy_predictions)
            input()

            assert biopsy_predictions.empty is not True, "The predictions should not be empty"
            assert ground_truth.empty is not True, "The ground truth should not be empty"

            for protein in ground_truth.columns:
                # calculate mae for all quartiles
                mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                    ground_truth=ground_truth, marker=protein, predictions=biopsy_predictions, std=2)

                quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
                    {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                     "Threshold": [quartiles[protein][0.25], quartiles[protein][0.5], quartiles[protein][0.75],
                                   quartiles[protein][0.75]], "Marker": protein,
                     "Biopsy": biopsy,
                     "Mode": mode,
                     "Std": 2
                     })])

    save_path = Path(save_path, "lgbm")
    if not save_path.exists():
        save_path.mkdir(parents=True)
        # save quartile performance
    quartile_performance.to_csv(Path(save_path, "quartile_performance.csv"))


def create_ae_quartile_performance(save_path: Path, imputation: str = None):
    logging.debug(f"Imputation method: {imputation}")
    if imputation is None:
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae", "predictions.csv"))
    elif imputation == "multi":
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae_m", "predictions.csv"))
    else:
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "vae", "predictions.csv"))
    quartile_performance = pd.DataFrame(columns=["MAE", "Quartile", "Marker", "Biopsy", "Mode"])

    # convert mode to upper cose in predictions
    predictions["Mode"] = predictions["Mode"].str.upper()

    for biopsy in predictions["Biopsy"].unique():
        for mode in predictions["Mode"].unique():
            biopsy_predictions = predictions[predictions["Biopsy"] == biopsy].copy()
            # select only exp
            biopsy_predictions = biopsy_predictions[biopsy_predictions["Mode"] == mode].copy()
            # select only FE  == 0
            if "FE" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["FE"] == 0].copy()
            # select no hyper
            if "HP" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["HP"] == 0].copy()

            if "Noise" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["Noise"] == 0].copy()

            if "Replace Value" in biopsy_predictions.columns:
                biopsy_predictions = biopsy_predictions[biopsy_predictions["Replace Value"] == "mean"].copy()

            biopsy_predictions.reset_index(drop=True, inplace=True)
            ground_truth: pd.DataFrame = load_test_data_set("exp", biopsy, 0)

            # remove biopsy mode FE and HP columns from biopsy predictions
            if "FE" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["FE"], inplace=True)
            if "HP" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["HP"], inplace=True)
            if "Noise" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Noise"], inplace=True)
            if "Mode" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Mode"], inplace=True)
            if "Biopsy" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Biopsy"], inplace=True)
            if "Replace Value" in biopsy_predictions.columns:
                biopsy_predictions.drop(columns=["Replace Value"], inplace=True)

            assert biopsy_predictions.empty is not True, "The predictions should not be empty"
            assert ground_truth.empty is not True, "The ground truth should not be empty"

            for protein in ground_truth.columns:
                # calculate mae for all quartiles
                mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                    ground_truth=ground_truth, marker=protein, predictions=biopsy_predictions, std=2)

                quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
                    {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                     "Threshold": [quartiles[protein][0.25], quartiles[protein][0.5], quartiles[protein][0.75],
                                   quartiles[protein][0.75]], "Marker": protein,
                     "Biopsy": biopsy,
                     "Mode": mode,
                     "Std": 2
                     })])

    # convert Std to int
    quartile_performance["Std"] = quartile_performance["Std"].astype(int)

    if imputation is None:
        save_path = Path(save_path, "ae")
    elif imputation == "multi":
        save_path = Path(save_path, "ae_m")
    else:
        save_path = Path(save_path, "vae")

    if not save_path.exists():
        save_path.mkdir(parents=True)
    # save quartile performance
    quartile_performance.to_csv(Path(save_path, "quartile_performance.csv"))


def create_en_quartile_performance(ground_truth: pd.DataFrame, predictions: pd.DataFrame, marker: str, biopsy: str,
                                   mode: str):
    # select biopsy and mode from ground truth
    ground_truth = ground_truth[(ground_truth["Biopsy"] == biopsy) & (ground_truth["Mode"] == mode)].copy()
    # select biopsy and mode from predictions
    predictions = predictions[(predictions["Biopsy"] == biopsy) & (predictions["Mode"] == mode)].copy()

    # calculate mae for all quartiles
    mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
        ground_truth=ground_truth, marker=marker, predictions=predictions, std=2)

    quartile_performance: pd.DataFrame = pd.concat([quartile_performance, pd.DataFrame(
        {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
         "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                       quartiles[marker][0.75]], "Marker": marker,
         "Biopsy": biopsy,
         "Mode": mode,
         "Std": 2
         })])


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
    setup_log_file(save_path=Path("quartile_performance"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", action="store", choices=["EN", "LGBM", "AE", "AE M", "VAE ALL"])
    args = parser.parse_args()

    model: str = args.model
    logging.debug(f"Model: {model}")

    save_path = Path("data/cleaned_data/quartile_performance")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    if model == "EN":
        pass
    elif model == "LGBM":
        create_lgbm_quartile_performance(save_path=save_path)

    elif model == "AE":
        create_ae_quartile_performance(save_path=save_path)
    elif model == "AE M":
        create_ae_quartile_performance(save_path=save_path, imputation="multi")
    elif model == "VAE ALL":
        create_ae_quartile_performance(save_path=save_path, imputation="vae")
