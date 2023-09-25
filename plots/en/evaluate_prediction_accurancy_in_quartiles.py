import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

save_path = Path("plots", "en", "quartiles")

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']
modes = ["ip", "exp"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric", choices=["MAE", "RMSE"], help="The metric", default="MAE")

    args = parser.parse_args()
    metric: str = args.metric

    # save_path = Path(save_path, mode)

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
                    str(Path("data", "tumor_mesmer", "preprocessed", f"{biopsy_name}_preprocessed_dataset.tsv")),
                    sep='\t')

                ground_truth = ground_truth[
                    np.abs(ground_truth[marker] - ground_truth[marker].mean()) <= (
                            2 * ground_truth[marker].std())].copy()

                # select all indexes of predictions which are in the ground truth index
                predictions = predictions.loc[ground_truth.index].copy()

                # extract the quartiles
                quartiles = ground_truth.quantile([0.25, 0.5, 0.75])
                # select the rows that are in the quartiles from the predictions and ground truth
                gt_quartile_1 = ground_truth[ground_truth[marker] <= quartiles[marker][0.25]]
                gt_quartile_2 = ground_truth[
                    (ground_truth[marker] > quartiles[marker][0.25]) & (ground_truth[marker] <= quartiles[marker][0.5])]
                gt_quartile_3 = ground_truth[
                    (ground_truth[marker] > quartiles[marker][0.5]) & (ground_truth[marker] <= quartiles[marker][0.75])]
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
                # mae_all = np.mean(np.abs(ground_truth[marker] - predictions["prediction"]))

                # add these values to a dataframe
                results.append(pd.DataFrame(
                    {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                     "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                   quartiles[marker][0.75]], "Marker": marker, "Biopsy": biopsy_name,
                     "Mode": mode}))

    results = pd.concat(results)
    results.to_csv(str(Path(save_path, f"accuracy.csv")), index=False)
    # change mode column to uppercase
    results["Mode"] = results["Mode"].str.upper()
    # plot the quartiles

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, palette={"IP": "lightblue", "EXP": "orange"})
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.box(False)
    # remove x and y label
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ax.set_title(f"Elastic Net \n{metric.upper()} per quartile\nAll Biopsies", fontsize=20, y=1.3)

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        # (("Q1", "IP"), ("Q1", "EXP")),
        # (("Q2", "IP"), ("Q2", "EXP")),
        # (("Q3", "IP"), ("Q3", "EXP")),
        # (("Q4", "IP"), ("Q4", "EXP")),
        # (("All", "IP"), ("All", "EXP")),
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        # (("Q4", "IP"), ("All", "IP")),
        (("Q1", "EXP"), ("Q2", "EXP")),
        (("Q2", "EXP"), ("Q3", "EXP")),
        (("Q3", "EXP"), ("Q4", "EXP")),
        # (("Q4", "EXP"), ("All", "EXP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(str(Path(save_path, f"accuracy.png")))
