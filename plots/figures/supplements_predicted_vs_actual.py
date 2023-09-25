import argparse, logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

save_path = Path("plots", "figures", "supplements", "predicted_vs_actual")
ground_truth_path = Path("data/cleaned_data/ground_truth")
predicted_path = Path("data/cleaned_data/predictions")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", help="the predicted marker", required=False)
    parser.add_argument("--biopsy", "-b", help="the biopsy used", required=True)
    parser.add_argument("--mode", choices=["ip", "exp"], default="exp", help="the mode used")
    parser.add_argument("--model", choices=["EN", "LGBM", "AE", "GNN", "AE M", "AE ALL"], help="the network used",
                        required=True)
    parser.add_argument("--spatial", "-sp", choices=[0, 23, 46, 92, 138, 184], default=0)
    args = parser.parse_args()
    marker: str = args.marker
    model: str = args.model
    biopsy: str = args.biopsy
    mode: str = str(args.mode).upper()
    spatial: int = args.spatial

    save_path = Path(save_path, model, biopsy, str(spatial))

    ground_truth_path: Path = Path(ground_truth_path, f"{biopsy}_preprocessed_dataset.tsv")
    predicted_path: Path = Path(predicted_path, model, "predictions.csv")

    ground_truth: pd.DataFrame = pd.read_csv(ground_truth_path, delimiter="\t", header=0)
    predictions: pd.DataFrame = pd.read_csv(predicted_path)
    # convert Mode to upper case
    predictions["Mode"] = predictions["Mode"].str.upper()
    # convert FE to int
    predictions["FE"] = predictions["FE"].astype(int)
    # convert HP to int
    predictions["HP"] = predictions["HP"].astype(int)

    if "Noise" in predictions.columns:
        # convert Noise to int
        predictions["Noise"] = predictions["Noise"].astype(int)

    # select only predictions for the given mode and biopsy from the predictions file and spatial
    predictions = predictions.loc[
        (predictions["Mode"] == mode.upper()) & (predictions["Biopsy"] == biopsy) & (
                predictions["FE"] == spatial)].reset_index(
        drop=True)

    # difference = ground_truth[marker] - predictions[marker]

    # fig = plt.figure(figsize=(5, 3), dpi=200)
    # plt.plot(difference, alpha=0.5, label=marker, marker='o', linestyle='None')
    # plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
    #         np.poly1d(np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(), 1))(
    #             np.unique(ground_truth[[marker]].values.flatten())), color='red')

    # plt.show()
    # plt.close('all')

    print(predictions)
    print(ground_truth)

    if marker is None:
        for marker in ground_truth.columns:
            fig = plt.figure(figsize=(5, 3), dpi=200)
            plt.scatter(ground_truth[marker], predictions[marker], alpha=0.5, label=marker)
            plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
                     np.poly1d(
                         np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(),
                                    1))(
                         np.unique(ground_truth[[marker]].values.flatten())), color='red')
            plt.xlabel(f"Ground Truth {marker}")
            plt.ylabel(f"Predicted {marker}")
            plt.tight_layout()

            if not save_path.exists():
                save_path.mkdir(parents=True)

            plt.savefig(Path(save_path, f"{marker}_predicted_vs_actual.png"), dpi=300)
    else:
        fig = plt.figure(figsize=(5, 3), dpi=200)
        plt.scatter(ground_truth[marker], predictions[marker], alpha=0.5, label=marker)
        plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
                 np.poly1d(
                     np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(),
                                1))(
                     np.unique(ground_truth[[marker]].values.flatten())), color='red')
        plt.xlabel(f"Ground Truth {marker}")
        plt.ylabel(f"Predicted {marker}")
        plt.tight_layout()

        if not save_path.exists():
            save_path.mkdir(parents=True)

        plt.savefig(Path(save_path, f"{marker}_predicted_vs_actual.png"), dpi=300)