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

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

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
    base_path: Path = Path("plots", "figures", "supplements", "ground_truth_distribution")

    if not base_path.exists():
        base_path.mkdir(parents=True)

    setup_log_file(save_path=base_path)

    for biopsy in BIOPSIES:
        patient: str = '_'.join(biopsy.split("_")[:2])

        save_path = Path(base_path, biopsy)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")

        variance_scores = []
        for protein in ground_truth.columns:
            gt = ground_truth[protein]
            # train = train_data[protein]

            sns.histplot(gt, color="blue", label="Expression", kde=True)
            # sns.histplot(train, color="green", label="TRAIN", kde=True)

            # change y axis label to cell count
            plt.ylabel("Cell Count")
            plt.xlabel(f"{protein} Expression")
            # plt.legend()
            plt.savefig(Path(save_path, f"{protein}.png"), dpi=300)
            # close figure
            plt.close()

        # plot violin plot for each biopsy
        fig = plt.figure(figsize=(10, 5), dpi=300)
        sns.violinplot(data=ground_truth)
        plt.title(biopsy.replace("_", " "))
        plt.savefig(Path(save_path, f"{biopsy}.png"), dpi=300)
        plt.close()
