import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import shutil

save_path = Path("plots", "en", "quartiles")

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']
modes = ["ip", "exp"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-b", "--biopsy", help="The test biopsy", required=True)
    #parser.add_argument("--mode", choices=["ip", "exp"], required=True, default="ip")

    args = parser.parse_args()
    #mode = args.mode

    #save_path = Path(save_path, mode)

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in modes:
        for biopsy_name in biopsies:
            if mode == "ip":
                load_path = Path("mesmer", "tumor_in_patient_en")
                if biopsy_name[-1] == "1":
                    train_biopsy = biopsy_name[:-1] + "2"
                else:
                    train_biopsy = biopsy_name[:-1] + "1"

            else:
                load_path = Path("mesmer", "tumor_exp_patient_en")
                train_biopsy = biopsy_name

            for marker in markers:
                predictions = pd.read_csv(
                    str(Path(load_path, train_biopsy, biopsy_name, marker, f"{marker}_predictions.csv")),
                    header=None)
                predictions = predictions.rename(columns={0: "prediction"})

                ground_truth = pd.read_csv(
                    str(Path("data", "tumor_mesmer", "preprocessed", f"{biopsy_name}_preprocessed_dataset.tsv")), sep='\t')

                # extract the quartiles
                quartiles = ground_truth.quantile([0.25, 0.5, 0.75])
                # select the rows that are in the quartiles from the predictions and ground truth
                gt_quartile_1 = ground_truth[ground_truth[marker] <= quartiles[marker][0.25]]
                gt_quartile_2 = ground_truth[
                    (ground_truth[marker] > quartiles[marker][0.25]) & (ground_truth[marker] <= quartiles[marker][0.5])]
                gt_quartile_3 = ground_truth[
                    (ground_truth[marker] > quartiles[marker][0.5]) & (ground_truth[marker] <= quartiles[marker][0.75])]
                gt_quartile_4 = ground_truth[ground_truth[marker] > quartiles[marker][0.75]]

                pred_quartile_1 = predictions.iloc[gt_quartile_1.index]
                pred_quartile_2 = predictions.iloc[gt_quartile_2.index]
                pred_quartile_3 = predictions.iloc[gt_quartile_3.index]
                pred_quartile_4 = predictions.iloc[gt_quartile_4.index]

                # Calculate MAE for all quartiles
                mae_1 = np.mean(np.abs(gt_quartile_1[marker] - pred_quartile_1["prediction"]))
                mae_2 = np.mean(np.abs(gt_quartile_2[marker] - pred_quartile_2["prediction"]))
                mae_3 = np.mean(np.abs(gt_quartile_3[marker] - pred_quartile_3["prediction"]))
                mae_4 = np.mean(np.abs(gt_quartile_4[marker] - pred_quartile_4["prediction"]))
                mae_all = np.mean(np.abs(ground_truth[marker] - predictions["prediction"]))

                # add these values to a dataframe
                results.append(pd.DataFrame(
                    {"MAE": [mae_1, mae_2, mae_3, mae_4, mae_all], "Quartile": ["Q1", "Q2", "Q3", "Q4", "All"],
                     "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                   quartiles[marker][0.75], "All"], "Marker": marker, "Biopsy": biopsy_name, "Mode": mode}))

    results = pd.concat(results)
    results.to_csv(str(Path(save_path, f"accuracy.csv")), index=False)

    # plot the quartiles

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y="MAE", hue="Mode", data=results, palette="Set2")
    ax.set_xlabel("Quartile")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE per quartile")
    plt.savefig(str(Path(save_path, f"accuracy.png")))
