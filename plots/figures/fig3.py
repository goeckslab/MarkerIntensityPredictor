import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str, save_path: Path):
    color_palette = {"0 µm": "grey", "23 µm": "magenta", "46 µm": "purple", "92 µm": "green", "138 µm": "yellow",
                     "184 µm": "blue"}
    dpi = 96
    if model == "GNN":
        fig = plt.figure(figsize=(800 / dpi, 350 / dpi), dpi=dpi)
    else:
        fig = plt.figure(figsize=(800 / dpi, 350 / dpi), dpi=dpi)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="FE", palette=color_palette)

    plt.ylabel("")
    plt.xlabel("")

    plt.box(False)
    # remove legend from fig
    plt.legend().set_visible(False)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.ylim(ylim[0], ylim[1])

    hue = "FE"
    hue_order = microns

    pairs = []
    for micron in microns:
        if micron == "0 µm" and model != "GNN":
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
        print(pairs)
        print(data["FE"].unique())
        raise

    plt.tight_layout()
    plt.savefig(Path(save_path), dpi=150)
    plt.close('all')


if __name__ == '__main__':
    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))

    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin([0, 23, 92, 184])]
    # select exp scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]

    # Add µm to the FE column
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # load ae scores
    ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
    # sort by markers
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    ae_scores = ae_scores[ae_scores["FE"].isin([0, 23, 92, 184])]
    # select exp scores
    ae_scores = ae_scores[ae_scores["Mode"] == "EXP"]
    # select mean replace value
    ae_scores = ae_scores[ae_scores["Replace Value"] == "mean"]
    # Add µm to the FE column
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    ae_scores = ae_scores[np.abs(ae_scores["MAE"] - ae_scores["MAE"].mean()) <= (3 * ae_scores["MAE"].std())]
    # ae_scores = ae_scores[np.abs(ae_scores["RMSE"] - ae_scores["RMSE"].mean()) <= (3 * ae_scores["RMSE"].std())]

    # load gnn scores
    gnn_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "gnn", "scores.csv"))
    # sort by markers
    gnn_scores.sort_values(by=["Marker"], inplace=True)
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    gnn_scores = gnn_scores[gnn_scores["FE"].isin([23, 46, 92, 184])]
    # select mean replace value
    gnn_scores = gnn_scores[gnn_scores["Replace Value"] == "mean"]
    # select exp scores
    gnn_scores = gnn_scores[gnn_scores["Mode"] == "EXP"]
    # Add µm to the FE column
    gnn_scores["FE"] = gnn_scores["FE"].astype(str) + " µm"
    gnn_scores["FE"] = pd.Categorical(gnn_scores['FE'], ["23 µm", "46 µm", "92 µm", "184 µm"])

    gnn_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    gnn_scores = gnn_scores[np.abs(gnn_scores["MAE"] - gnn_scores["MAE"].mean()) <= (3 * gnn_scores["MAE"].std())]
    gnn_scores = gnn_scores[
        np.abs(gnn_scores["RMSE"] - gnn_scores["RMSE"].mean()) <= (3 * gnn_scores["RMSE"].std())]

    # load image from images fig3 folder
    spatial_information_image = plt.imread(Path("images", "fig3", "Panel_A.png"))
    gnn_spatial_information_image = plt.imread(Path("images", "fig3", "Panel_D.png"))
    test_image = plt.imread(Path("images", "fig3", "img.png"))

    image_path = Path("plots", "figures", "fig3_subplots")
    if not image_path.exists():
        image_path.mkdir(parents=True)

    create_boxen_plot(data=gnn_scores, metric="MAE", ylim=[0, 0.5],
                      microns=["23 µm", "46 µm", "92 µm", "184 µm"], model="GNN",
                      save_path=Path(image_path, "gnn_results.png"))
    create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                      microns=["0 µm", "23 µm", "92 µm", "184 µm"], model="LGBM",
                      save_path=Path(image_path, "lgbm_results.png"))
    create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0, 0.5],
                      microns=["0 µm", "23 µm", "92 µm", "184 µm"], model="AE",
                      save_path=Path(image_path, "ae_results.png"))

    # load gnn image
    gnn_results = plt.imread(Path(image_path, "gnn_results.png"))
    # load lgbm image
    lgbm_results = plt.imread(Path(image_path, "lgbm_results.png"))
    # load ae image
    ae_results = plt.imread(Path(image_path, "ae_results.png"))

    dpi = 96
    # Create new figure
    fig = plt.figure(figsize=(1400 / dpi, 1000 / dpi), dpi=dpi)
    gspec = fig.add_gridspec(6, 2)

    # ax = fig.add_subplot(gspec[:, :3])
    # plt.box(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.imshow(test_image, aspect="auto")

    ax1 = fig.add_subplot(gspec[:3, 0])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.1, 1.1, "A", transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    # show spatial information image
    ax1.imshow(spatial_information_image)

    ax2 = fig.add_subplot(gspec[3:, 0])
    # remove box from ax2
    plt.box(False)
    # remove ticks from ax2
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(-0.1, 1.1, "C", transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    # show gnn spatial information image
    ax2.imshow(gnn_spatial_information_image)

    ax3 = fig.add_subplot(gspec[0:2, 1])
    ax3.set_title('LGBM 0 vs. 15 µm, 60 µm and 120 µm', rotation='horizontal', fontsize=20)
    ax3.text(0, 1.1, "B", transform=ax3.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(lgbm_results, aspect='equal')

    ax4 = fig.add_subplot(gspec[2:4, 1])
    ax4.set_title('AE 0 vs. 15 µm, 60 µm and 120 µm', rotation='horizontal', fontsize=20)
    ax4.text(0, 1.1, "D", transform=ax4.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(ae_results)

    ax5 = fig.add_subplot(gspec[4:, 1])
    ax5.set_title('GNN 15 µm vs. 30 µm, 60 µm and 120 µm', rotation='horizontal', fontsize=20)
    ax5.text(0, 1.1, "E", transform=ax5.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    # remove box from ax5
    plt.box(False)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(gnn_results)

    plt.tight_layout()
    plt.savefig(Path("plots", "figures", "fig3.png"), dpi=250)
