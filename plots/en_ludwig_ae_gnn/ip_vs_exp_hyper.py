import os, sys, argparse, shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List
from statannotations.Annotator import Annotator
import numpy as np

save_path = Path("plots/")


def load_scores(load_path: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if Path(name).suffix == ".csv":
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, "Not all biopsies have been processed"
    scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
    return scores


def load_ae_scores(mode: str, replace_value: str):
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == 0]
    all_scores = all_scores[all_scores["HP"] == 1]
    return all_scores


def create_boxen_plot(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
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

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("-n", "--network", choices=["Ludwig", "AE"], help="Network to be plotted",
                        default="Ludwig")

    args = parser.parse_args()
    markers = args.markers
    network = args.network

    save_path = Path(save_path)
    save_path = Path(save_path, network, "ip_vs_exp_hyper")
    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if network == "Ludwig":
        # Load scores
        ip_scores = load_scores("data/scores/Mesmer/ip/Ludwig_hyper")
        ip_scores["Mode"] = "IP"
        exp_scores = load_scores("data/scores/Mesmer/exp/Ludwig_hyper")
        exp_scores["Mode"] = "EXP"

    else:
        ip_scores = load_ae_scores(replace_value="mean", mode="ip")
        ip_scores["Mode"] = "IP"
        exp_scores = load_ae_scores(replace_value="mean", mode="exp")
        exp_scores["Mode"] = "EXP"

    # Combine ip and exp scores
    scores: pd.DataFrame = pd.concat([ip_scores, exp_scores], ignore_index=True)

    if network == "AE":
        # remove outliers greater than 3 std for the MAE column
        scores = scores[np.abs(scores["MAE"] - scores["MAE"].mean()) <= (3 * scores["MAE"].std())]
        # remove outliers greater than 3 std for the RMSE column
        scores = scores[np.abs(scores["RMSE"] - scores["RMSE"].mean()) <= (3 * scores["RMSE"].std())]

    create_boxen_plot(data=scores, metric="MAE", title="MAE", save_folder=save_path, file_name="MAE", ylim=[0, 1])
    create_boxen_plot(data=scores, metric="RMSE", title="RMSE", save_folder=save_path, file_name="RMSE", ylim=[0, 1])
