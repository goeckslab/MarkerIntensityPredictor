import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

save_path = Path("cleaned_data/images")
ground_truth_path = Path("data/cleaned_data/ground_truth")
predicted_path = Path("data/cleaned_data/predictions")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", help="the predicted marker", required=True)
    parser.add_argument("--biopsy", "-b", help="the biopsy used", required=True)
    parser.add_argument("--mode", choices=["ip", "exp"], default="ip", help="the mode used")
    parser.add_argument("--network", choices=["EN", "LGBM", "AE", "GNN", "AE M", "AE ALL"], help="the network used", required=True)
    args = parser.parse_args()
    marker: str = args.marker
    network: str = args.network
    biopsy: str = args.biopsy
    mode: str = str(args.mode).upper()

    ground_truth_path: Path = Path(ground_truth_path, f"{biopsy}_preprocessed_dataset.tsv")
    predicted_path: Path = Path(predicted_path, network, "predictions.csv")

    ground_truth: pd.DataFrame = pd.read_csv(ground_truth_path, delimiter="\t", header=0)
    predictions: pd.DataFrame = pd.read_csv(predicted_path)
    predictions.rename(columns={0: marker}, inplace=True)

    # select only predictions for the given mode and biopsy from the predictions file
    predictions = predictions.loc[(predictions["Mode"] == mode) & (predictions["Biopsy"] == biopsy)].reset_index(
        drop=True)

    tested: str = Path(ground_truth_path).stem
    test_splits: [] = tested.split("_")
    if test_splits[2] == 2:
        tested = " ".join(test_splits[:2]) + " 1"
        trained = " ".join(test_splits[:2]) + " 2"
    else:
        tested = " ".join(test_splits[:2]) + " 2"
        trained = " ".join(test_splits[:2]) + " 1"

    difference = ground_truth[marker] - predictions[marker]

    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(difference, alpha=0.5, label=marker, marker='o', linestyle='None')
    # plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
    #         np.poly1d(np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(), 1))(
    #             np.unique(ground_truth[[marker]].values.flatten())), color='red')

    plt.show()
    plt.close('all')

    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(ground_truth[marker], predictions[marker], alpha=0.5, label=marker)
    plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
             np.poly1d(np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(), 1))(
                 np.unique(ground_truth[[marker]].values.flatten())), color='red')
    plt.xlabel(f"Ground Truth {marker}")
    plt.ylabel(f"Predicted {marker}")

    if not save_path.exists():
        save_path.mkdir(parents=True)
        
    plt.savefig(Path(save_path, f"{marker}_{network}_{mode}_{biopsy}_predicted_vs_actual.png"), dpi=300)
