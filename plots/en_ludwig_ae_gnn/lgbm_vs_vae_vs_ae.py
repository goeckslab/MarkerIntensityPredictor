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

save_path = Path("plots/en_ludwig_ae_gnn/lgbm_vs_vae_vs_ae")


def create_boxen_plot(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                      ylim: List):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="Type", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["LGBM", "AE", "VAE"],
                       palette={"AE": "green", "VAE": "red", "LGBM": "orange"})

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
    hue_order = ["LGBM", "AE", "VAE"]
    pairs = [
        (("pRB", "VAE"), ("pRB", "LGBM")),
        (("CD45", "VAE"), ("CD45", "LGBM")),
        (("CK19", "VAE"), ("CK19", "LGBM")),
        (("Ki67", "VAE"), ("Ki67", "LGBM")),
        (("aSMA", "VAE"), ("aSMA", "LGBM")),
        (("Ecad", "VAE"), ("Ecad", "LGBM")),
        (("PR", "VAE"), ("PR", "LGBM")),
        (("CK14", "VAE"), ("CK14", "LGBM")),
        (("HER2", "VAE"), ("HER2", "LGBM")),
        (("AR", "VAE"), ("AR", "LGBM")),
        (("CK17", "VAE"), ("CK17", "LGBM")),
        (("p21", "VAE"), ("p21", "LGBM")),
        (("Vimentin", "VAE"), ("Vimentin", "LGBM")),
        (("pERK", "VAE"), ("pERK", "LGBM")),
        (("EGFR", "VAE"), ("EGFR", "LGBM")),
        (("ER", "VAE"), ("ER", "LGBM")),

        (("pRB", "VAE"), ("pRB", "AE")),
        (("CD45", "VAE"), ("CD45", "AE")),
        (("CK19", "VAE"), ("CK19", "AE")),
        (("Ki67", "VAE"), ("Ki67", "AE")),
        (("aSMA", "VAE"), ("aSMA", "AE")),
        (("Ecad", "VAE"), ("Ecad", "AE")),
        (("PR", "VAE"), ("PR", "AE")),
        (("CK14", "VAE"), ("CK14", "AE")),
        (("HER2", "VAE"), ("HER2", "AE")),
        (("AR", "VAE"), ("AR", "AE")),
        (("CK17", "VAE"), ("CK17", "AE")),
        (("p21", "VAE"), ("p21", "AE")),
        (("Vimentin", "VAE"), ("Vimentin", "AE")),
        (("pERK", "VAE"), ("pERK", "AE")),
        (("EGFR", "VAE"), ("EGFR", "AE")),
        (("ER", "VAE"), ("ER", "AE")),

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


def load_lgbm_scores(mode: str, spatial: int):
    mode = mode.upper()
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_ae_scores(mode: str, replace_value: str, spatial: int):
    mode = mode.upper()
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_vae_scores(mode: str, replace_value: str, spatial: int):
    mode = mode.upper()
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "vae", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", choices=["ip", "exp"], type=str, default="ip")
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

    if mode == 'ip':
        vae_scores = load_vae_scores(mode="ip", replace_value=replace_value, spatial=0)
        ae_scores = load_ae_scores(mode="ip", replace_value=replace_value, spatial=0)
        lgbm_scores = load_lgbm_scores(mode="ip", spatial=0)
    elif mode == 'exp':
        vae_scores = load_vae_scores(mode="exp", replace_value=replace_value, spatial=0)
        ae_scores = load_ae_scores(mode="exp", replace_value=replace_value, spatial=0)
        lgbm_scores = load_lgbm_scores(mode="exp", spatial=0)
    else:
        raise ValueError("Mode not recognized")

    # combine ae and fe scores
    scores = pd.concat([lgbm_scores, ae_scores, vae_scores], axis=0)

    # duplicate each row in scores
    # scores = pd.concat(scores, ignore_index=True)

    create_boxen_plot(data=scores, metric="MAE", title="MAE", save_folder=save_path,
                      file_name="MAE", ylim=[0, 0.6])
    create_boxen_plot(data=scores, metric="RMSE", title="RMSE", save_folder=save_path,
                      file_name="RMSE", ylim=[0, 0.6])
