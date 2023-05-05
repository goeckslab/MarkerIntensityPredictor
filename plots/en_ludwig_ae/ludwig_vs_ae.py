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

save_path = Path("plots/en_ludwig_ae/ludwig_vs_ae")


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
    # plt.legend().set_visible(False)

    hue = "Mode"
    hue_order = ["Ludwig", "AE"]
    pairs = [
        (("pRB", "Ludwig"), ("pRB", "AE")),
        (("CD45", "Ludwig"), ("CD45", "AE")),
        (("CK19", "Ludwig"), ("CK19", "AE")),
        (("Ki67", "Ludwig"), ("Ki67", "AE")),
        (("aSMA", "Ludwig"), ("aSMA", "AE")),
        (("Ecad", "Ludwig"), ("Ecad", "AE")),
        (("PR", "Ludwig"), ("PR", "AE")),
        (("CK14", "Ludwig"), ("CK14", "AE")),
        (("HER2", "Ludwig"), ("HER2", "AE")),
        (("AR", "Ludwig"), ("AR", "AE")),
        (("CK17", "Ludwig"), ("CK17", "AE")),
        (("p21", "Ludwig"), ("p21", "AE")),
        (("Vimentin", "Ludwig"), ("Vimentin", "AE")),
        (("pERK", "Ludwig"), ("pERK", "AE")),
        (("EGFR", "Ludwig"), ("EGFR", "AE")),
        (("ER", "Ludwig"), ("ER", "AE")),
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
    print(load_path)
    scores = []
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if "scores.csv" in name:
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, "Not all biopsies have been processed"

    return pd.concat(scores)


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", choices=["ip", "op", "exp"], type=str, default="ip")
    parser.add_argument("-rv", "--replace_value", choices=["mean", "zero"], default="mean")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    parser.add_argument("--metric", choices=["mae", "rmse"], default="mae")

    args = parser.parse_args()
    mode = args.mode
    markers = args.markers
    add_noise = "noise" if args.an else "no_noise"
    replace_value = args.replace_value
    metric: str = args.metric

    save_path = Path(save_path, mode)
    save_path = Path(save_path, replace_value)
    save_path = Path(save_path, add_noise)

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    print(mode)
    if mode == 'ip':
        ludwig_scores = load_scores(f"data/scores/Mesmer/in_patient/Ludwig")
        ae_scores = load_scores(f"ae_imputation/ip/{replace_value}/{add_noise}")
    elif mode == 'op':
        ludwig_scores = load_scores(f"data/scores/Mesmer/out_patient/Ludwig")
        ae_scores = load_scores(f"ae_imputation/op/{replace_value}/{add_noise}")
    elif mode == 'exp':
        ludwig_scores = load_scores(f"data/scores/Mesmer/exp/Ludwig")
        ae_scores = load_scores(f"ae_imputation/exp/{replace_value}/{add_noise}")
    else:
        raise ValueError("Mode not recognized")

    # Select best performing iteration per marker
    ae_scores = ae_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_scores = ae_scores.groupby(["Marker", "Biopsy"]).first().reset_index()
    ae_scores["Mode"] = "AE"

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Type", "Mode"]]
    ae_scores = ae_scores[["Marker", "MAE", "RMSE", "Biopsy", "Type", "Mode"]]

    # combine ae and fe scores
    scores = pd.concat([ae_scores, ludwig_scores], axis=0)
    # duplicate each row in scores
    scores = pd.concat([scores] * 30, ignore_index=True)

    create_boxen_plot_per_segmentation(data=scores, metric=metric.upper(), title=metric.upper(), save_folder=save_path,
                                       file_name=metric.upper(),
                                       ylim=[0, 0.6])