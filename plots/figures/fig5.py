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


def create_boxen_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List) -> plt.Figure:
    hue = "Network"
    x = "FE"
    order = ["0 µm", "15 µm", "60 µm", "120 µm"]
    hue_order = ["LGBM", "AE", "AE M"]
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, order=order,
                       palette={"EN": "purple", "LGBM": "green", "AE": "grey", "AE M": "darkgrey"})

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # set y ticks of fig
    plt.box(False)
    # remove legend from fig
    plt.legend(prop={"size": 7}, loc='upper center')
    # plt.legend().set_visible(False)

    pairs = [
        (("0 µm", "LGBM"), ("0 µm", "AE")),
        (("0 µm", "LGBM"), ("0 µm", "AE M")),
        (("0 µm", "AE"), ("0 µm", "AE M")),
        (("15 µm", "LGBM"), ("15 µm", "AE")),
        (("15 µm", "LGBM"), ("15 µm", "AE M")),
        (("15 µm", "AE"), ("15 µm", "AE M")),
        (("60 µm", "LGBM"), ("60 µm", "AE")),
        (("60 µm", "LGBM"), ("60 µm", "AE M")),
        (("60 µm", "AE"), ("60 µm", "AE M")),
        (("120 µm", "LGBM"), ("120 µm", "AE")),
        (("120 µm", "LGBM"), ("120 µm", "AE M")),
        (("120 µm", "AE"), ("120 µm", "AE M")),
    ]

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str, legend_position: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "blue"}

    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette, showfliers=False)

    plt.ylabel("")
    plt.xlabel("")

    plt.box(False)
    plt.legend(bbox_to_anchor=legend_position, loc='center', fontsize=7, ncol=2)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.ylim(ylim[0], ylim[1])

    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue

        if micron == "15 µm" and model == "GNN":
            continue
        # Create pairs of (micron, 0 µm) for each marker
        for marker in data["Marker"].unique():
            if model == "GNN":
                pairs.append(((marker, micron), (marker, "15 µm")))
            else:
                pairs.append(((marker, micron), (marker, "0 µm")))

    try:
        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21',
                 'Vimentin', 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                              hide_non_significant=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2, line_height=0.01)
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

    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(["0 µm", "15 µm", "60 µm", "120 µm"])
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
    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_m_scores["FE"] = ae_m_scores["FE"].cat.rename_categories(["0 µm", "15 µm", "60 µm", "120 µm"])
    # sort by marker and FE
    ae_m_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    ae_m_scores = ae_m_scores[np.abs(ae_m_scores["MAE"] - ae_m_scores["MAE"].mean()) <= (3 * ae_m_scores["MAE"].std())]
    ae_m_scores = ae_m_scores[
        np.abs(ae_m_scores["RMSE"] - ae_m_scores["RMSE"].mean()) <= (3 * ae_m_scores["RMSE"].std())]

    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin([0, 23, 92, 184])]
    # select exp scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]
    # only select non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    # Add µm to the FE column
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])

    # rename 23 to 15, 92 to 60 and 184 to 120
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(["0 µm", "15 µm", "60 µm", "120 µm"])

    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    dpi = 300
    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('AE S 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1.text(-0.01, 1.3, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)

    # ax3 = ax3.imshow(lgbm_results)
    ax1 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0, 0.6],
                            microns=["0 µm", "15 µm", "60 µm", "120 µm"], model="AE", legend_position=[0.1, 0.8])

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('AE M 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax2.text(x=-0.01, y=1.3, s="b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    # ax4.imshow(ae_results)
    ax2 = create_boxen_plot(data=ae_m_scores, metric="MAE", ylim=[0, 0.6],
                            microns=["0 µm", "15 µm", "60 µm", "120 µm"], model="AE M", legend_position=[0.1, 0.8])

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.text(x=-0.01, y=1.3, s="c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('Performance', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax3 = create_boxen_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.8])

    plt.tight_layout()
    plt.savefig(Path(save_path, "fig5.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(save_path, "fig5.eps"), dpi=300, bbox_inches='tight', format='eps')

    print("Mean and std of MAE scores per network")

    print(all_scores.groupby(["Network", "FE"])["MAE"].agg(["mean", "std"]))
    # print count of scores per network and fe
    print(all_scores.groupby(["Network", "FE"])["MAE"].count())
