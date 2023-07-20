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
    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    en_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]

    # load image from images fig2 folder
    train_test_split = plt.imread(Path("images", "fig2", "train_test_split.png"))

    # ax1 = fig.add_subplot(2, 2, 1)
    # fig.add_subplot(2, 2, 3).set_title("223")
    # fig.add_subplot(1, 2, 2).set_title("122")

    fig = plt.figure(figsize=(10, 7), dpi=150)
    gspec = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gspec[0, :1])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # add image to figure
    ax1.imshow(train_test_split, aspect='auto')

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('Elastic Net', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax2 = create_boxen_plot(data=en_scores, metric="MAE", ylim=[0.0, 0.4])

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.text(-0.1, 1.15, "C", transform=ax3.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('LBGM', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax3 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0.0, 0.4])

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(Path("plots", "figures", "fig2.png"), dpi=300)
    sys.exit()
