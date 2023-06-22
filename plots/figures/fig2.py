import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator


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
    # mae_all = np.mean(np.abs(ground_truth[marker] - predictions["prediction"]))

    return mae_1, mae_2, mae_3, mae_4, quartiles


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List) -> plt.Figure:
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Mode", palette={"IP": "lightblue", "EXP": "orange"})

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    # set y ticks of fig
    plt.box(False)
    # remove legend from fig
    plt.legend().set_visible(False)

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        (("pRB", "IP"), ("pRB", "EXP")),
        (("CD45", "IP"), ("CD45", "EXP")),
        (("CK19", "IP"), ("CK19", "EXP")),
        (("Ki67", "IP"), ("Ki67", "EXP")),
        (("aSMA", "IP"), ("aSMA", "EXP")),
        (("Ecad", "IP"), ("Ecad", "EXP")),
        (("PR", "IP"), ("PR", "EXP")),
        (("CK14", "IP"), ("CK14", "EXP")),
        (("HER2", "IP"), ("HER2", "EXP")),
        (("AR", "IP"), ("AR", "EXP")),
        (("CK17", "IP"), ("CK17", "EXP")),
        (("p21", "IP"), ("p21", "EXP")),
        (("Vimentin", "IP"), ("Vimentin", "EXP")),
        (("pERK", "IP"), ("pERK", "EXP")),
        (("EGFR", "IP"), ("EGFR", "EXP")),
        (("ER", "IP"), ("ER", "EXP")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


def create_boxen_plot_ip_vs_exp(results: pd.DataFrame, metric: str, title: str):
    # plot the quartiles
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, hue_order=["IP", "EXP"],
                       palette={"IP": "lightblue", "EXP": "orange"})
    ax.set_xlabel("Quartile")

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        (("Q1", "EXP"), ("Q2", "EXP")),
        (("Q2", "EXP"), ("Q3", "EXP")),
        (("Q3", "EXP"), ("Q4", "EXP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    en_scores = pd.read_csv(Path("data", "cleaned_scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    en_scores = pd.concat([en_scores] * 30)

    lgbm_scores = pd.read_csv(Path("data", "cleaned_scores", "lgbm", "scores.csv"))
    ae_scores = pd.read_csv(Path("data", "cleaned_scores", "ae", "scores.csv"))

    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[(ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    # load image from images fig2 folder
    image = plt.imread(Path("images", "fig2", "Fig2 Label 2.png"))

    # ax1 = fig.add_subplot(2, 2, 1)
    # fig.add_subplot(2, 2, 3).set_title("223")
    # fig.add_subplot(1, 2, 2).set_title("122")

    fig = plt.figure(figsize=(10, 7), dpi=150)
    gspec = fig.add_gridspec(4, 3)
    ax1 = fig.add_subplot(gspec[0, :])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # add image to figure
    ax1.imshow(image)

    ax2 = fig.add_subplot(gspec[1, :2])
    ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('Elastic Net', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax2 = create_boxen_plot(data=en_scores, metric="MAE", ylim=[0.0, 0.4])

    ax3 = fig.add_subplot(gspec[1, 2])
    ax3.text(-0.1, 1.15, "C", transform=ax3.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)

    ax4 = fig.add_subplot(gspec[2, :2])
    ax4.text(-0.1, 1.15, "D", transform=ax4.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax4.set_title('LBGM', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax4 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0.0, 0.4])

    ax5 = fig.add_subplot(gspec[2, 2])
    ax5.text(-0.1, 1.15, "E", transform=ax5.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)

    ax6 = fig.add_subplot(gspec[3, :2])
    ax6.text(-0.1, 1.15, "F", transform=ax6.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax6.set_title('AE', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax6 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0.0, 0.4])

    ax7 = fig.add_subplot(gspec[3, 2])
    ax7.text(-0.1, 1.15, "G", transform=ax7.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)

    # add suptitle
    # fig.suptitle("Figure 2", fontsize=16, fontweight='bold', rotation='vertical', x=-0.01)
    plt.tight_layout()

    # fig.add_subplot(2, 2, 3).set_title("223")
    # fig.add_subplot(1, 2, 2).set_title("122")

    plt.show()
    sys.exit()

    print("Loading scores...")

    print(ae_scores)
    input()
