import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, show_legend: bool = False) -> plt.Figure:
    hue = "Mode"
    hue_order = ["IP", "AP"]
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, hue_order=hue_order,
                       palette={"IP": "lightblue", "AP": "orange"})

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

    plt.legend(bbox_to_anchor=[0.7, 0.9], loc='center', ncol=2)

    pairs = [
        (("pRB", "IP"), ("pRB", "AP")),
        (("CD45", "IP"), ("CD45", "AP")),
        (("CK19", "IP"), ("CK19", "AP")),
        (("Ki67", "IP"), ("Ki67", "AP")),
        (("aSMA", "IP"), ("aSMA", "AP")),
        (("Ecad", "IP"), ("Ecad", "AP")),
        (("PR", "IP"), ("PR", "AP")),
        (("CK14", "IP"), ("CK14", "AP")),
        (("HER2", "IP"), ("HER2", "AP")),
        (("AR", "IP"), ("AR", "AP")),
        (("CK17", "IP"), ("CK17", "AP")),
        (("p21", "IP"), ("p21", "AP")),
        (("Vimentin", "IP"), ("Vimentin", "AP")),
        (("pERK", "IP"), ("pERK", "AP")),
        (("EGFR", "IP"), ("EGFR", "AP")),
        (("ER", "IP"), ("ER", "AP")),
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
    hue_order = ["LGBM", "EN", "AE", "AE M"]
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, order=order,
                       palette={"EN": "purple", "LGBM": "green", "AE": "grey", "AE M": "darkgrey",
                                "AE ALL": "lightgrey"})

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
        (("IP", "LGBM"), ("IP", "EN")),
        (("IP", "LGBM"), ("IP", "AE")),
        (("IP", "LGBM"), ("IP", "AE M")),
        (("IP", "AE"), ("IP", "AE M")),
        (("AP", "LGBM"), ("AP", "EN")),
        (("AP", "LGBM"), ("AP", "AE")),
        (("AP", "LGBM"), ("AP", "AE M")),
        (("AP", "AE"), ("AP", "AE M")),
    ]

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    # replace EXP WITH AP
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})

    en_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    # replace EXP WITH AP
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})

    ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
    ae_m_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_m", "scores.csv"))
    # replace EXP WITH AP
    ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[
        (ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    # select only non hp scores
    ae_scores = ae_scores[ae_scores["HP"] == 0]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_m_scores = ae_m_scores[
        (ae_m_scores["FE"] == 0) & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0)]
    # select only non hp scores
    ae_m_scores = ae_m_scores[ae_m_scores["HP"] == 0]
    ae_m_scores.sort_values(by=["Marker"], inplace=True)
    # replace EXP WITH AP
    ae_m_scores["Mode"] = ae_m_scores["Mode"].replace({"EXP": "AP"})

    # assert that FE column only contains 0
    assert (lgbm_scores["FE"] == 0).all(), "FE column should only contain 0 for lgbm_scores"
    assert (ae_m_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_m_scores"
    assert (ae_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_scores"

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, en_scores, ae_scores, ae_m_scores], axis=0)



    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value", "Hyper"], inplace=True)
    # rename MOde AP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # load image from images fig2 folder
    # image = plt.imread(Path("images", "fig3", "ae_workflow_2.png"))

    fig = plt.figure(figsize=(10, 8), dpi=300)
    gspec = fig.add_gridspec(6, 3)

    ax1 = fig.add_subplot(gspec[:2, :])
    ax1.text(-0.05, 1.15, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title("AE (Single Protein)", rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax1 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0.0, 0.8])

    ax2 = fig.add_subplot(gspec[2:4, :])
    ax2.text(-0.05, 1.1, "b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('AE (Multi Protein)', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax2 = create_boxen_plot(data=ae_m_scores, metric="MAE", ylim=[0.0, 0.8], show_legend=True)

    ax3 = fig.add_subplot(gspec[4:6, :2])
    ax3.text(-0.08, 1.1, "c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('Performance', rotation='vertical', x=-0.08, y=0, fontsize=12)
    ax3 = create_boxen_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.8])


    plt.tight_layout()

    # plt.show()

    # save figure
    fig.savefig(Path("images", "fig3", "fig3.png"), dpi=300, bbox_inches='tight')
    fig.savefig(Path("images", "fig3", "fig3.eps"), dpi=300, bbox_inches='tight', format='eps')
    # print mean and std of all scores per network of MAE scores
    print("Mean and std of MAE scores per network")
    print(all_scores.groupby(["Network"])["MAE"].agg(["mean", "std"]))
    sys.exit()
