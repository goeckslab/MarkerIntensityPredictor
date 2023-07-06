import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path
import argparse
from typing import List
import shutil
from statannotations.Annotator import Annotator

# Create violin plots showing the difference between feature engineering and no feature engineering
# Specific for a single spatial distance

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]

save_path = Path("plots/ae/fe_vs_no_fe")


def create_boxen_plot(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                      ylim: List, color_palette, spatial_distance: str):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="FE", split=True, cut=0, palette=color_palette)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="FE", palette=color_palette)

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    # set y ticks of fig
    if args.markers:
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)
        ax.set_xticklabels(x_ticks, rotation=0, fontsize=20)
    plt.box(False)
    # remove legend from fig
    # plt.legend().set_visible(False)

    hue = "FE"
    hue_order = [spatial_distance, "0 µm"]
    pairs = [
        (("pRB", spatial_distance), ("pRB", "0 µm")),
        (("CD45", spatial_distance), ("CD45", "0 µm")),
        (("CK19", spatial_distance), ("CK19", "0 µm")),
        (("Ki67", spatial_distance), ("Ki67", "0 µm")),
        (("aSMA", spatial_distance), ("aSMA", "0 µm")),
        (("Ecad", spatial_distance), ("Ecad", "0 µm")),
        (("PR", spatial_distance), ("PR", "0 µm")),
        (("CK14", spatial_distance), ("CK14", "0 µm")),
        (("HER2", spatial_distance), ("HER2", "0 µm")),
        (("AR", spatial_distance), ("AR", "0 µm")),
        (("CK17", spatial_distance), ("CK17", "0 µm")),
        (("p21", spatial_distance), ("p21", "0 µm")),
        (("Vimentin", spatial_distance), ("Vimentin", "0 µm")),
        (("pERK", spatial_distance), ("pERK", "0 µm")),
        (("EGFR", spatial_distance), ("EGFR", "0 µm")),
        (("ER", spatial_distance), ("ER", "0 µm")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_boxen_plot_all_spatial(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                                  ylim: List, color_palette):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="FE", split=True, cut=0, palette=color_palette)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="FE", palette=color_palette)

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    # set y ticks of fig
    if args.markers:
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)
        ax.set_xticklabels(x_ticks, rotation=0, fontsize=20)
    plt.box(False)
    # remove legend from fig
    # plt.legend().set_visible(False)

    hue = "FE"
    hue_order = ["0 µm", "23 µm", "92 µm", "184 µm"]
    pairs = [
        (("pRB", "23 µm"), ("pRB", "0 µm")),
        (("CD45", "23 µm"), ("CD45", "0 µm")),
        (("CK19", "23 µm"), ("CK19", "0 µm")),
        (("Ki67", "23 µm"), ("Ki67", "0 µm")),
        (("aSMA", "23 µm"), ("aSMA", "0 µm")),
        (("Ecad", "23 µm"), ("Ecad", "0 µm")),
        (("PR", "23 µm"), ("PR", "0 µm")),
        (("CK14", "23 µm"), ("CK14", "0 µm")),
        (("HER2", "23 µm"), ("HER2", "0 µm")),
        (("AR", "23 µm"), ("AR", "0 µm")),
        (("CK17", "23 µm"), ("CK17", "0 µm")),
        (("p21", "23 µm"), ("p21", "0 µm")),
        (("Vimentin", "23 µm"), ("Vimentin", "0 µm")),
        (("pERK", "23 µm"), ("pERK", "0 µm")),
        (("EGFR", "23 µm"), ("EGFR", "0 µm")),
        (("ER", "23 µm"), ("ER", "0 µm")),
        # (("pRB", "46 µm"), ("pRB", "0 µm")),
        # (("CD45", "46 µm"), ("CD45", "0 µm")),
        # (("CK19", "46 µm"), ("CK19", "0 µm")),
        # (("Ki67", "46 µm"), ("Ki67", "0 µm")),
        # (("aSMA", "46 µm"), ("aSMA", "0 µm")),
        # (("Ecad", "46 µm"), ("Ecad", "0 µm")),
        # (("PR", "46 µm"), ("PR", "0 µm")),
        # (("CK14", "46 µm"), ("CK14", "0 µm")),
        # (("HER2", "46 µm"), ("HER2", "0 µm")),
        # (("AR", "46 µm"), ("AR", "0 µm")),
        # (("CK17", "46 µm"), ("CK17", "0 µm")),
        # (("p21", "46 µm"), ("p21", "0 µm")),
        # (("Vimentin", "46 µm"), ("Vimentin", "0 µm")),
        # (("pERK", "46 µm"), ("pERK", "0 µm")),
        # (("EGFR", "46 µm"), ("EGFR", "0 µm")),
        # (("ER", "46 µm"), ("ER", "0 µm")),
        (("pRB", "92 µm"), ("pRB", "0 µm")),
        (("CD45", "92 µm"), ("CD45", "0 µm")),
        (("CK19", "92 µm"), ("CK19", "0 µm")),
        (("Ki67", "92 µm"), ("Ki67", "0 µm")),
        (("aSMA", "92 µm"), ("aSMA", "0 µm")),
        (("Ecad", "92 µm"), ("Ecad", "0 µm")),
        (("PR", "92 µm"), ("PR", "0 µm")),
        (("CK14", "92 µm"), ("CK14", "0 µm")),
        (("HER2", "92 µm"), ("HER2", "0 µm")),
        (("AR", "92 µm"), ("AR", "0 µm")),
        (("CK17", "92 µm"), ("CK17", "0 µm")),
        (("p21", "92 µm"), ("p21", "0 µm")),
        (("Vimentin", "92 µm"), ("Vimentin", "0 µm")),
        (("pERK", "92 µm"), ("pERK", "0 µm")),
        (("EGFR", "92 µm"), ("EGFR", "0 µm")),
        (("ER", "92 µm"), ("ER", "0 µm")),
        # (("pRB", "138 µm"), ("pRB", "0 µm")),
        # (("CD45", "138 µm"), ("CD45", "0 µm")),
        # (("CK19", "138 µm"), ("CK19", "0 µm")),
        # (("Ki67", "138 µm"), ("Ki67", "0 µm")),
        # (("aSMA", "138 µm"), ("aSMA", "0 µm")),
        # (("Ecad", "138 µm"), ("Ecad", "0 µm")),
        # (("PR", "138 µm"), ("PR", "0 µm")),
        # (("CK14", "138 µm"), ("CK14", "0 µm")),
        # (("HER2", "138 µm"), ("HER2", "0 µm")),
        # (("AR", "138 µm"), ("AR", "0 µm")),
        # (("CK17", "138 µm"), ("CK17", "0 µm")),
        # (("p21", "138 µm"), ("p21", "0 µm")),
        # (("Vimentin", "138 µm"), ("Vimentin", "0 µm")),
        # (("pERK", "138 µm"), ("pERK", "0 µm")),
        # (("EGFR", "138 µm"), ("EGFR", "0 µm")),
        # (("ER", "138 µm"), ("ER", "0 µm")),
        (("pRB", "184 µm"), ("pRB", "0 µm")),
        (("CD45", "184 µm"), ("CD45", "0 µm")),
        (("CK19", "184 µm"), ("CK19", "0 µm")),
        (("Ki67", "184 µm"), ("Ki67", "0 µm")),
        (("aSMA", "184 µm"), ("aSMA", "0 µm")),
        (("Ecad", "184 µm"), ("Ecad", "0 µm")),
        (("PR", "184 µm"), ("PR", "0 µm")),
        (("CK14", "184 µm"), ("CK14", "0 µm")),
        (("HER2", "184 µm"), ("HER2", "0 µm")),
        (("AR", "184 µm"), ("AR", "0 µm")),
        (("CK17", "184 µm"), ("CK17", "0 µm")),
        (("p21", "184 µm"), ("p21", "0 µm")),
        (("Vimentin", "184 µm"), ("Vimentin", "0 µm")),
        (("pERK", "184 µm"), ("pERK", "0 µm")),
        (("EGFR", "184 µm"), ("EGFR", "0 µm")),
        (("ER", "184 µm"), ("ER", "0 µm")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_line_plot(data: pd.DataFrame, metric: str):
    fig = plt.figure(dpi=200, figsize=(10, 6))
    ax = sns.lineplot(x="Marker", y=metric, hue="FE", data=data)
    plt.show()


def load_ae_scores(mode: str, replace_value: str, add_noise: str, hyper: int):
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["HP"] == hyper]
    return all_scores


if __name__ == '__main__':
    # argsparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("-sp", "--spatial", type=int, help="Spatial distance", default=46,
                        choices=[23, 46, 92, 138, 184])
    parser.add_argument("--mode", type=str, default="ip", choices=["ip", "exp"])
    parser.add_argument("-hp", "--hyper", action="store_true", default=False)
    args = parser.parse_args()

    # load mesmer mae scores from data mesmer folder and all subfolders
    spatial_distance = args.spatial
    mode: str = args.mode
    markers = args.markers
    hyper = args.hyper

    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial_distance))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    print(args.spatial)

    ae_scores = load_ae_scores(mode=mode, replace_value="mean", add_noise="no_noise",
                               hyper=hyper)
    base_scores = ae_scores[ae_scores["FE"] == 0]
    base_scores = base_scores.groupby(["Marker", "Biopsy", "Experiment", "Replace Value", "Noise", "HP"]).head(5)
    # calculate mean of MAE scores
    base_scores = base_scores.groupby(["Marker", "Biopsy", "Experiment", "Replace Value", "Noise", "HP"]).mean().reset_index()

    compare_scores = ae_scores[ae_scores["FE"] == spatial_distance]
    compare_scores = compare_scores.groupby(["Marker", "Biopsy", "Experiment"]).head(5)
    compare_scores = compare_scores.groupby(["Marker", "Biopsy", "Experiment"]).mean().reset_index()

    my_pal = {"0 µm": "grey", "23 µm": "magenta", "46 µm": "purple", "92 µm": "green", "138 µm": "yellow",
              "184 µm": "blue"}

    # combine both dataframes
    scores = pd.concat([base_scores, compare_scores], axis=0)
    # clip scores by 3 stds
    scores = scores[scores["MAE"] < scores["MAE"].mean() + 3 * scores["MAE"].std()]

    scores["FE"] = scores["FE"].fillna(0)
    # convert fe to int
    scores["FE"] = scores["FE"].astype(int)

    # rename sp_23 to 23 µm
    scores["FE"] = scores["FE"].replace(
        {0: "0 µm", 23: "23 µm", 46: "46 µm", 92: "92 µm", 138: "138 µm", 184: "184 µm"})

    ae_scores["FE"] = ae_scores["FE"].replace(
        {0: "0 µm", 23: "23 µm", 46: "46 µm", 92: "92 µm", 138: "138 µm", 184: "184 µm"})

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # Create bar plot which compares in patient performance of the different segementations for each biopsy
    # The bar plot should be saved in the plots folder

    if args.markers:
        y_lim = [0, 0.3]
    else:
        y_lim = [0, 0.5]

    scores = scores[scores["MAE"] < scores["MAE"].mean() + 3 * scores["MAE"].std()]
    scores = scores[scores["RMSE"] < scores["RMSE"].mean() + 3 * scores["RMSE"].std()]
    create_boxen_plot(data=scores, metric="MAE",
                      title=f"In & EXP patient performance using spatial feature engineering",
                      file_name="MAE", save_folder=save_path, ylim=y_lim, color_palette=my_pal,
                      spatial_distance=f"{spatial_distance} µm")

    create_boxen_plot(data=scores, metric="RMSE",
                      title=f"In & EXP patient performance using spatial feature engineering",
                      file_name="RMSE", save_folder=save_path, ylim=y_lim, color_palette=my_pal,
                      spatial_distance=f"{spatial_distance} µm")

    # slect only 23, 92 and 184
    scores = ae_scores[ae_scores["FE"].isin(["0 µm", "23 µm", "92 µm", "184 µm"])]

    # sort by FE, first 0µm, then 23µm, then 92µm, then 184µm
    scores["FE"] = pd.Categorical(scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    scores.sort_values(by=["FE"], inplace=True)

    # remove 3 stds for MAE and RMSE
    scores = scores[scores["MAE"] < scores["MAE"].mean() + 3 * scores["MAE"].std()]
    scores = scores[scores["RMSE"] < scores["RMSE"].mean() + 3 * scores["RMSE"].std()]

    create_boxen_plot_all_spatial(data=scores, metric="MAE",
                                  title=f"In & EXP patient performance using spatial feature engineering",
                                  file_name="MAE_all", save_folder=save_path, ylim=y_lim, color_palette=my_pal)
