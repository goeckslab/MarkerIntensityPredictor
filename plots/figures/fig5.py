import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator
import logging

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("plots/figures/fig5.log"),
                        logging.StreamHandler()
                    ])


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str):
    color_palette = {"0 µm": "grey", "23 µm": "magenta", "46 µm": "purple", "92 µm": "green", "138 µm": "yellow",
                     "184 µm": "blue"}

    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    plt.ylabel("")
    plt.xlabel("")

    plt.box(False)
    # remove legend from fig
    plt.legend().set_visible(False)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.ylim(ylim[0], ylim[1])

    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue

        if micron == "23 µm" and model == "GNN":
            continue
        # Create pairs of (micron, 0 µm) for each marker
        for marker in data["Marker"].unique():
            if model == "GNN":
                pairs.append(((marker, micron), (marker, "23 µm")))
            else:
                pairs.append(((marker, micron), (marker, "0 µm")))

    try:
        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21',
                 'Vimentin',
                 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                              hide_non_significant=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
        annotator.apply_and_annotate()

    except:
        logging.error(f"Model: {model}")
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise

    # plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return ax


if __name__ == '__main__':
    save_path = Path("images", "fig5")

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # load ae scores
    ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))

    # sort by markers
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    ae_scores = ae_scores[ae_scores["FE"].isin([0, 23, 92, 184])]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_scores = ae_scores[
        (ae_scores["Mode"] == "EXP") & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0) & (
                ae_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # load ae multi scores
    ae_m_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_m", "scores.csv"))

    # sort by markers
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    ae_m_scores = ae_m_scores[ae_m_scores["FE"].isin([0, 23, 92, 184])]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_m_scores = ae_m_scores[
        (ae_m_scores["Mode"] == "EXP") & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0) & (
                ae_m_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_m_scores["FE"] = ae_m_scores["FE"].astype(str) + " µm"
    ae_m_scores["FE"] = pd.Categorical(ae_m_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    # sort by marker and FE
    ae_m_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    ae_m_scores = ae_m_scores[np.abs(ae_m_scores["MAE"] - ae_m_scores["MAE"].mean()) <= (3 * ae_m_scores["MAE"].std())]
    # ae_m_scores = ae_m_scores[np.abs(ae_m_scores["RMSE"] - ae_m_scores["RMSE"].mean()) <= (3 * ae_m_scores["RMSE"].std())]

    gnn_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "gnn", "scores.csv"))

    print(len(gnn_scores["Marker"].unique()))
    input()


    # sort by markers
    gnn_scores.sort_values(by=["Marker"], inplace=True)
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    gnn_scores = gnn_scores[gnn_scores["FE"].isin([23, 46, 92, 184])]
    # select mean replace value
    gnn_scores = gnn_scores[gnn_scores["Replace Value"] == "mean"]
    # select exp scores
    gnn_scores = gnn_scores[gnn_scores["Mode"] == "EXP"]
    # select no noise
    #gnn_scores = gnn_scores[gnn_scores["Noise"] == 0]
    # select no hp
    #gnn_scores = gnn_scores[gnn_scores["HP"] == 0]

    print(gnn_scores)
    # print unique biopsy value
    print(gnn_scores["Biopsy"].unique())
    print(gnn_scores["Marker"].unique())

    # calculate amount of markers per biopsy per experiemtn
    print(gnn_scores.groupby(["Biopsy", "FE"]).count()["Marker"])

    # select same amount of markers per biopsy per FE




    # Add µm to the FE column
    gnn_scores["FE"] = gnn_scores["FE"].astype(str) + " µm"
    gnn_scores["FE"] = pd.Categorical(gnn_scores['FE'], ["23 µm", "46 µm", "92 µm", "184 µm"])
    gnn_scores.sort_values(by=["Marker", "FE"], inplace=True)



    dpi = 300
    cm = 1 / 2.54  # centimeters in inches
    # Create new figure
    fig = plt.figure(figsize=(22 * cm, 18 * cm), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('AE 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.1, y=0, fontsize=7)
    ax1.text(-0.05, 1.1, "A", transform=ax1.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)

    # ax3 = ax3.imshow(lgbm_results)
    ax1 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0, 0.5],
                            microns=["0 µm", "23 µm", "92 µm", "184 µm"], model="AE")

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('AE M 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.1, y=0, fontsize=7)
    ax2.text(-0.05, 1.1, "B", transform=ax2.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    # ax4.imshow(ae_results)
    ax2 = create_boxen_plot(data=ae_m_scores, metric="MAE", ylim=[0, 0.5],
                            microns=["0 µm", "23 µm", "92 µm", "184 µm"], model="AE M")

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.set_title('GNN 15µm vs. 46µm, 60 µm and 120 µm', rotation='vertical', x=-0.1, y=0, fontsize=7)
    ax3.text(-0.05, 1.1, "C", transform=ax3.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    # ax4.imshow(ae_results)


    ax3 = create_boxen_plot(data=gnn_scores, metric="MAE", ylim=[0, 0.5],
                            microns=["23 µm", "46 µm", "92 µm", "184 µm"], model="GNN")

    plt.tight_layout()
    plt.savefig(Path(save_path, "fig5.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(save_path, "fig5.eps"), dpi=300, bbox_inches='tight', format='eps')
