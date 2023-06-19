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

ludwig_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig")
sp_23_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_23")
sp_46_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_46")
sp_92_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_92")
sp_138_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_138")
sp_184_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_184")

ludwig_op_source_path = Path("data/scores/Mesmer/exp/Ludwig")
sp_23_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_23")
sp_46_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_46")
sp_92_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_92")
sp_138_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_138")
sp_184_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_184")

save_path = Path("plots/ludwig/fe_vs_no_fe")


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
    hue_order = ["0 µm", "23 µm", "92 µm",  "184 µm"]
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


def load_scores(source_path: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(source_path):
        for name in files:
            if Path(name).suffix == ".csv":
                print("Loading file: ", os.path.join(root, name))
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, f"Should have loaded 8 biopsies but loaded: {len(scores)}, for path {source_path}"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':
    # argsparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("-sp", "--spatial", type=int, help="Spatial distance", default=46,
                        choices=[23, 46, 92, 138, 184])
    parser.add_argument("--mode", type=str, default="ip", choices=["ip", "exp"])
    args = parser.parse_args()

    # load mesmer mae scores from data mesmer folder and all subfolders
    spatial_distance = args.spatial
    mode: str = args.mode
    markers = args.markers

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
    if mode == "ip":
        if args.spatial == 23:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_23_ip_source_path))
            my_pal = {"0 µm": "grey", "23 µm": "yellow"}
        elif args.spatial == 46:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_46_ip_source_path))
            my_pal = {"0 µm": "grey", "46 µm": "purple"}
        elif args.spatial == 92:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_92_ip_source_path))
            my_pal = {"0 µm": "grey", "92 µm": "green"}
        elif args.spatial == 138:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_138_ip_source_path))
            my_pal = {"0 µm": "grey", "138 µm": "orange"}
        elif args.spatial == 184:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_184_ip_source_path))
            my_pal = {"0 µm": "grey", "184 µm": "blue"}
        else:
            raise ValueError("Spatial distance not supported")
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_ip_source_path))
    else:
        if args.spatial == 23:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_23_op_source_path))
            my_pal = {"0 µm": "grey", "23 µm": "yellow"}
        elif args.spatial == 46:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_46_op_source_path))
            my_pal = {"0 µm": "grey", "46 µm": "purple"}
        elif args.spatial == 92:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_92_op_source_path))
            my_pal = {"0 µm": "grey", "92 µm": "green"}
        elif args.spatial == 138:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_138_op_source_path))
            my_pal = {"0 µm": "grey", "138 µm": "orange"}
        elif args.spatial == 184:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_184_op_source_path))
            my_pal = {"0 µm": "grey", "184 µm": "blue"}
        else:
            raise ValueError("Spatial distance not supported")
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_op_source_path))

    # combine both dataframes
    scores = pd.concat([fe_scores, no_fe_scores], axis=0)
    # duplicate each row in scores
    # scores = pd.concat([scores] * 30, ignore_index=True)
    # convert Nan to 0
    scores["FE"] = scores["FE"].fillna(0)
    # convert fe to int
    scores["FE"] = scores["FE"].astype(int)

    # rename sp_23 to 23 µm
    scores["FE"] = scores["FE"].replace(
        {0: "0 µm", 23: "23 µm", 46: "46 µm", 92: "92 µm", 138: "138 µm", 184: "184 µm"})

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # Create bar plot which compares in patient performance of the different segementations for each biopsy
    # The bar plot should be saved in the plots folder

    if args.markers:
        y_lim = [0, 0.3]
    else:
        y_lim = [0, 0.6]

    create_boxen_plot(data=scores, metric="MAE",
                      title=f"In & EXP patient performance using spatial feature engineering",
                      file_name="MAE", save_folder=save_path, ylim=y_lim, color_palette=my_pal,
                      spatial_distance=f"{spatial_distance} µm")

    create_boxen_plot(data=scores, metric="RMSE",
                      title=f"In & EXP patient performance using spatial feature engineering",
                      file_name="RMSE", save_folder=save_path, ylim=y_lim, color_palette=my_pal,
                      spatial_distance=f"{spatial_distance} µm")

    # load scores for 0, 23, 92 and 138 µm
    all_scores: List = [load_scores(source_path=str(sp_92_op_source_path)),
                        load_scores(source_path=str(sp_23_op_source_path)),
                        load_scores(source_path=str(sp_184_op_source_path)),
                        load_scores(source_path=str(ludwig_op_source_path))]
    # combine scores
    scores = pd.concat(all_scores, axis=0)

    scores["FE"] = scores["FE"].replace(
        {0: "0 µm", 23: "23 µm", 46: "46 µm", 92: "92 µm", 138: "138 µm", 184: "184 µm"})

    scores["FE"] = pd.Categorical(scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])
    scores.sort_values(by=["FE"], inplace=True)

    my_pal = {"0 µm": "grey", "23 µm": "magenta", "46 µm": "purple", "92 µm": "green", "138 µm": "yellow",
              "184 µm": "blue"}

    create_boxen_plot_all_spatial(data=scores, metric="MAE",
                                  title=f"In & EXP patient performance using spatial feature engineering",
                                  file_name="MAE_all", save_folder=save_path, ylim=y_lim, color_palette=my_pal)
