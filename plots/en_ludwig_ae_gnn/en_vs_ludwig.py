import os, shutil, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from statannotations.Annotator import Annotator
import numpy as np

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

save_path = Path("plots/en_ludwig_ae_gnn/en_vs_ludwig")


def create_boxen_plot(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                      ylim: List):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="Type", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network")

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

    hue = "Network"
    hue_order = ["EN", "Light GBM"]
    pairs = [
        (("pRB", "EN"), ("pRB", "Light GBM")),
        (("CD45", "EN"), ("CD45", "Light GBM")),
        (("CK19", "EN"), ("CK19", "Light GBM")),
        (("Ki67", "EN"), ("Ki67", "Light GBM")),
        (("aSMA", "EN"), ("aSMA", "Light GBM")),
        (("Ecad", "EN"), ("Ecad", "Light GBM")),
        (("PR", "EN"), ("PR", "Light GBM")),
        (("CK14", "EN"), ("CK14", "Light GBM")),
        (("HER2", "EN"), ("HER2", "Light GBM")),
        (("AR", "EN"), ("AR", "Light GBM")),
        (("CK17", "EN"), ("CK17", "Light GBM")),
        (("p21", "EN"), ("p21", "Light GBM")),
        (("Vimentin", "EN"), ("Vimentin", "Light GBM")),
        (("pERK", "EN"), ("pERK", "Light GBM")),
        (("EGFR", "EN"), ("EGFR", "Light GBM")),
        (("ER", "EN"), ("ER", "Light GBM")),
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


def load_scores(load_path: str):
    scores = []
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if Path(name).suffix == ".csv":
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, "Not all biopsies have been processed"

    return scores


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", choices=["ip", "exp"], type=str, default="ip")

    args = parser.parse_args()
    mode = args.mode
    markers = args.markers

    save_path = Path(save_path, mode)
    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    en_scores = load_scores(f"data/scores/Mesmer/{'ip' if mode == 'ip' else 'exp'}/EN")
    ludwig_scores = load_scores(f"data/scores/Mesmer/{'ip' if mode == 'ip' else 'exp'}/Ludwig")

    en_scores = pd.concat(en_scores, ignore_index=True)

    # rename mode to network
    en_scores = en_scores.rename(columns={"Mode": "Network"})

    ludwig_scores = pd.concat(ludwig_scores, ignore_index=True)
    # rename mode to network
    ludwig_scores = ludwig_scores.rename(columns={"Mode": "Network"})
    # rename ludwig to light gbm
    ludwig_scores["Network"] = "Light GBM"

    # duplicate rows in en scores
    en_scores = pd.concat([en_scores] * 30, ignore_index=True, axis=0)

    # combine en and ludwig scores
    scores = pd.concat([en_scores, ludwig_scores], ignore_index=True, axis=0)
    # scores = pd.concat([scores] * 30, ignore_index=True)

    create_boxen_plot(data=scores, metric="MAE", title="MAE", save_folder=save_path,
                      file_name="MAE",
                      ylim=[0, 0.6])

    create_boxen_plot(data=scores, metric="RMSE", title="RMSE", save_folder=save_path,
                      file_name="RMSE",
                      ylim=[0, 0.6])
