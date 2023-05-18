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

save_path = Path("plots/en_ludwig_ae/en_vs_ludwig")


def create_boxen_plot_per_segmentation(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                                       ylim: List):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="Type", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Mode")

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
    #plt.legend().set_visible(False)

    hue = "Mode"
    hue_order = ["EN", "Ludwig"]
    pairs = [
        (("pRB", "EN"), ("pRB", "Ludwig")),
        (("CD45", "EN"), ("CD45", "Ludwig")),
        (("CK19", "EN"), ("CK19", "Ludwig")),
        (("Ki67", "EN"), ("Ki67", "Ludwig")),
        (("aSMA", "EN"), ("aSMA", "Ludwig")),
        (("Ecad", "EN"), ("Ecad", "Ludwig")),
        (("PR", "EN"), ("PR", "Ludwig")),
        (("CK14", "EN"), ("CK14", "Ludwig")),
        (("HER2", "EN"), ("HER2", "Ludwig")),
        (("AR", "EN"), ("AR", "Ludwig")),
        (("CK17", "EN"), ("CK17", "Ludwig")),
        (("p21", "EN"), ("p21", "Ludwig")),
        (("Vimentin", "EN"), ("Vimentin", "Ludwig")),
        (("pERK", "EN"), ("pERK", "Ludwig")),
        (("EGFR", "EN"), ("EGFR", "Ludwig")),
        (("ER", "EN"), ("ER", "Ludwig")),
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
    parser.add_argument("--mode", choices=["ip", "op", "exp"], type=str, default="ip")

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

    save_path.mkdir(parents=True, exist_ok=True)

    en_scores = load_scores(f"data/scores/Mesmer/{'in_patient' if mode == 'ip' else 'out_patient'}/EN")
    ludwig_scores = load_scores(f"data/scores/Mesmer/{'in_patient' if mode == 'ip' else 'out_patient'}/Ludwig")

    # concat en_score & ludwig_scores to one dataframe
    scores = pd.concat(en_scores + ludwig_scores, ignore_index=True)

    # duplicate each row in scores
    scores = pd.concat([scores] * 30, ignore_index=True)

    print(scores)

    create_boxen_plot_per_segmentation(data=scores, metric="MAE", title="MAE", save_folder=save_path, file_name="MAE",
                                       ylim=[0, 0.6])
