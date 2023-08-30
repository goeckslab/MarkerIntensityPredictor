import sys
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging
from scipy.stats import ks_2samp
from sklearn.metrics import explained_variance_score

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ae_imputation_m/debug.log"),
                        logging.StreamHandler()
                    ])


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
    save_path: Path = Path("plots", "figures", "supplements", "prediction_distribution")

    argparse = ArgumentParser()
    argparse.add_argument("--biopsy", "-b", help="the biopsy used. Should be just 9_2_1", required=True)
    argparse.add_argument("-sp", "--spatial", action="store", help="The spatial radius used",
                          choices=[0, 23, 46, 92, 138, 184], type=int, default=0)
    # argparse.add_argument("--mode", choices=["ip", "exp"], default="ip", help="the mode used")
    argparse.add_argument("--model", choices=["EN", "LGBM", "AE", "GNN", "AE M", "VAE ALL"], help="the model used",
                          required=True)
    args = argparse.parse_args()

    biopsy: str = args.biopsy
    # mode: str = args.mode
    model: str = args.model
    spatial: int = args.spatial
    patient: str = '_'.join(biopsy.split("_")[:2])

    save_path = Path(save_path, model, biopsy, str(spatial))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    setup_log_file(save_path=save_path)

    logging.debug(f"Biopsy: {biopsy}")
    logging.debug(f"Model: {model}")
    logging.debug(f"Patient: {patient}")
    logging.debug(f"Spatial: {spatial}")

    assert patient in biopsy, "The biopsy should be of the form 9_2_1, where 9_2 is the patient and 1 is the biopsy. Patient should be in biopsy"

    if model == "LGBM":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "lgbm", "predictions.csv"),
                                                sep=",")

        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    elif model == "EN":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "en", "predictions.csv"),
                                                sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    elif model == "AE":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae", "predictions.csv"),
                                                sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    elif model == "GNN":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "gnn", "predictions.csv"),
                                                sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    elif model == "AE M":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae_m", "predictions.csv"),
                                                sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    elif model == "VAE ALL":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "predictions", "vae_all", "predictions.csv"),
            sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")

    else:
        raise ValueError("Model not recognized")

    # select only 9_2_1 biopsy predictions from predictions
    predictions = predictions[predictions["Biopsy"] == biopsy]

    # convert mode to upper
    predictions["Mode"] = predictions["Mode"].str.upper()
    # select only EXP predictions.
    predictions = predictions[predictions["Mode"] == "EXP"]

    # convert noise fe and hp to int
    if "Noise" in predictions.columns:
        predictions["Noise"] = predictions["Noise"].astype(int)
    if "FE" in predictions.columns:
        predictions["FE"] = predictions["FE"].astype(int)

    if "HP" in predictions.columns:
        predictions["HP"] = predictions["HP"].astype(int)

    if "HP" in predictions.columns:
        predictions = predictions[predictions["HP"] == 0]

    if "Noise" in predictions.columns:
        predictions = predictions[predictions["Noise"] == 0]

    if "FE" in predictions.columns:
        predictions = predictions[predictions["FE"] == spatial]

    if "Replace Value" in predictions.columns:
        if model == "VAE ALL":
            predictions = predictions[predictions["Replace Value"] == "zero"]
        else:
            predictions = predictions[predictions["Replace Value"] == "mean"]

    # if columns, value, hp, noise and fe are present drop them from predictions df
    if "Replace Value" in predictions.columns:
        # drop column
        predictions = predictions.drop(columns=["Replace Value"])

    if "HP" in predictions.columns:
        # drop column
        predictions = predictions.drop(columns=["HP"])

    if "Noise" in predictions.columns:
        # drop column
        predictions = predictions.drop(columns=["Noise"])

    if "FE" in predictions.columns:
        # drop column
        predictions = predictions.drop(columns=["FE"])

    # remove biopsy and mode columns
    predictions: pd.DataFrame = predictions.drop(columns=["Mode", "Biopsy"])

    variance_scores = []
    for protein in predictions.columns:
        pred = predictions[protein]
        gt = ground_truth[protein]
        # train = train_data[protein]

        result = ks_2samp(gt, pred)

        print(f"{protein}: {result.pvalue}")

        # add variance score to variance scores list
        variance_scores.append({
            "Protein": protein,
            "Variance Score": explained_variance_score(gt, pred)
        })

        sns.histplot(pred, color="orange", label="Predicted", kde=True)
        # scale y-axis of gt and train to match pred

        sns.histplot(gt, color="blue", label="Ground Truth", kde=True)
        # sns.histplot(train, color="green", label="TRAIN", kde=True)

        # change y axis label to cell count
        plt.ylabel("Cell Count")
        plt.xlabel(f"{protein} Expression")
        plt.legend()
        plt.savefig(Path(save_path, f"{protein}.png"), dpi=300, bbox_inches='tight')
        plt.close('all')

    # convert variance scores to df
    variance_scores = pd.DataFrame(variance_scores)
    # sort df by variance score
    variance_scores = variance_scores.sort_values(by="Variance Score", ascending=False)
    # save df
    variance_scores.to_csv(Path(save_path, "variance_scores.csv"), sep=",", index=False)
