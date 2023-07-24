import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator


def create_boxen_plot_ip_vs_exp_quartile(data: pd.DataFrame, metric: str) -> plt.Figure:
    # plot the quartiles
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=data, hue_order=["IP", "EXP"],
                       palette={"IP": "lightblue", "EXP": "orange"})
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())

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
    annotator = Annotator(ax, pairs, data=data, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    return ax


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


def create_boxen_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List) -> plt.Figure:
    hue = "Network"
    x = "Mode"
    order = ["IP", "AP"]
    hue_order = ["EN", "LGBM", "AE", "AE ALL"]
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, order=order,
                       palette={"EN": "purple", "LGBM": "green", "AE": "grey", "AE ALL": "lightgrey"})

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
    plt.legend().set_visible(False)

    pairs = [
        (("IP", "EN"), ("IP", "LGBM")),
        (("IP", "LGBM"), ("IP", "AE")),
        (("IP", "AE"), ("IP", "AE ALL")),
        (("AP", "EN"), ("AP", "LGBM")),
        (("AP", "LGBM"), ("AP", "AE")),
        (("AP", "AE"), ("AP", "AE ALL")),
    ]

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, order=order, hue=hue, hue_order=hue_order,
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
    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    en_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]

    ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
    ae_all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_all", "scores.csv"))

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[(ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    # select only non hp scores
    ae_scores = ae_scores[ae_scores["HP"] == 0]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_all_scores = ae_all_scores[
        (ae_all_scores["FE"] == 0) & (ae_all_scores["Replace Value"] == "mean") & (ae_all_scores["Noise"] == 0)]
    # select only non hp scores
    ae_all_scores = ae_all_scores[ae_all_scores["HP"] == 0]
    ae_all_scores.sort_values(by=["Marker"], inplace=True)

    # assert that FE column only contains 0
    assert (lgbm_scores["FE"] == 0).all(), "FE column should only contain 0 for lgbm_scores"
    assert (ae_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_scores"
    assert (ae_all_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_all_scores"

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, en_scores, ae_scores, ae_all_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value", "Hyper"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # load image from images fig2 folder
    image = plt.imread(Path("images", "fig3", "ae_workflow.png"))

    dpi = 96
    fig = plt.figure(figsize=(10, 9), dpi=dpi)
    gspec = fig.add_gridspec(4, 3)
    ax1 = fig.add_subplot(gspec[0, :2])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')

    # add image to figure
    ax1.imshow(image, aspect='auto')

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title("AE (Single Marker)", rotation='vertical', x=-0.1, y=-0.2, fontsize=7)
    ax2 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0.0, 0.8])

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.text(-0.1, 1.15, "C", transform=ax3.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('AE (All Markers)', rotation='vertical', x=-0.1, y=0, fontsize=7)
    ax3 = create_boxen_plot(data=ae_all_scores, metric="MAE", ylim=[0.0, 0.8])

    ax4 = fig.add_subplot(gspec[3, :2])
    ax4.text(-0.15, 1.15, "D", transform=ax4.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax4.set_title('Performance', rotation='vertical', x=-0.15, y=0.1, fontsize=7)
    ax4 = create_boxen_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.8])

    # add suptitle
    # fig.suptitle("Figure 2", fontsize=16, fontweight='bold', rotation='vertical', x=-0.01)
    plt.tight_layout()

    # plt.show()

    # save figure
    fig.savefig(Path("images", "fig3", "fig3.png"), dpi=300, bbox_inches='tight')
    sys.exit()
