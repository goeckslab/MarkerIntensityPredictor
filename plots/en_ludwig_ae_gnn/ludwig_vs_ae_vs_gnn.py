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

save_path = Path("plots/en_ludwig_ae_gnn/ludwig_vs_ae_vs_gnn")


def create_boxen_plot(data: pd.DataFrame, metric: str, title: str, save_folder: Path, file_name: str,
                      ylim: List):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="Type", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["Light GBM", "AE", "GNN"])

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
    hue_order = ["Light GBM", "AE", "GNN"]
    pairs = [
        (("pRB", "Light GBM"), ("pRB", "AE")),
        (("CD45", "Light GBM"), ("CD45", "AE")),
        (("CK19", "Light GBM"), ("CK19", "AE")),
        (("Ki67", "Light GBM"), ("Ki67", "AE")),
        (("aSMA", "Light GBM"), ("aSMA", "AE")),
        (("Ecad", "Light GBM"), ("Ecad", "AE")),
        (("PR", "Light GBM"), ("PR", "AE")),
        (("CK14", "Light GBM"), ("CK14", "AE")),
        (("HER2", "Light GBM"), ("HER2", "AE")),
        (("AR", "Light GBM"), ("AR", "AE")),
        (("CK17", "Light GBM"), ("CK17", "AE")),
        (("p21", "Light GBM"), ("p21", "AE")),
        (("Vimentin", "Light GBM"), ("Vimentin", "AE")),
        (("pERK", "Light GBM"), ("pERK", "AE")),
        (("EGFR", "Light GBM"), ("EGFR", "AE")),
        (("ER", "Light GBM"), ("ER", "AE")),
        (("pRB", "Light GBM"), ("pRB", "GNN")),
        (("CD45", "Light GBM"), ("CD45", "GNN")),
        (("CK19", "Light GBM"), ("CK19", "GNN")),
        (("Ki67", "Light GBM"), ("Ki67", "GNN")),
        (("aSMA", "Light GBM"), ("aSMA", "GNN")),
        (("Ecad", "Light GBM"), ("Ecad", "GNN")),
        (("PR", "Light GBM"), ("PR", "GNN")),
        (("CK14", "Light GBM"), ("CK14", "GNN")),
        (("HER2", "Light GBM"), ("HER2", "GNN")),
        (("AR", "Light GBM"), ("AR", "GNN")),
        (("CK17", "Light GBM"), ("CK17", "GNN")),
        (("p21", "Light GBM"), ("p21", "GNN")),
        (("Vimentin", "Light GBM"), ("Vimentin", "GNN")),
        (("pERK", "Light GBM"), ("pERK", "GNN")),
        (("EGFR", "Light GBM"), ("EGFR", "GNN")),
        (("ER", "Light GBM"), ("ER", "GNN")),
        (("pRB", "AE"), ("pRB", "GNN")),
        (("CD45", "AE"), ("CD45", "GNN")),
        (("CK19", "AE"), ("CK19", "GNN")),
        (("Ki67", "AE"), ("Ki67", "GNN")),
        (("aSMA", "AE"), ("aSMA", "GNN")),
        (("Ecad", "AE"), ("Ecad", "GNN")),
        (("PR", "AE"), ("PR", "GNN")),
        (("CK14", "AE"), ("CK14", "GNN")),
        (("HER2", "AE"), ("HER2", "GNN")),
        (("AR", "AE"), ("AR", "GNN")),
        (("CK17", "AE"), ("CK17", "GNN")),
        (("p21", "AE"), ("p21", "GNN")),
        (("Vimentin", "AE"), ("Vimentin", "GNN")),
        (("pERK", "AE"), ("pERK", "GNN")),
        (("EGFR", "AE"), ("EGFR", "GNN")),
        (("ER", "AE"), ("ER", "GNN")),

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


def load_ae_scores(mode: str, replace_value: str, add_noise: str, spatial: int):
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_gnn_scores(mode: str, replace_value: str, spatial: int):
    all_scores = pd.read_csv(Path("data", "scores", "gnn", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["FE"] == spatial]
    return all_scores


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", choices=["ip", "exp"], type=str, default="ip")
    parser.add_argument("-rv", "--replace_value", choices=["mean", "zero"], default="mean")
    parser.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    parser.add_argument("-sp", "--spatial", choices=[23, 46, 92, 138, 184], default=23, type=int)

    args = parser.parse_args()
    mode = args.mode
    markers = args.markers
    replace_value = args.replace_value
    metric: str = args.metric
    spatial = args.spatial

    save_path = Path(save_path, mode)
    save_path = Path(save_path, replace_value)
    save_path = Path(save_path, f"{spatial}")

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    print(mode)
    if mode == 'ip':
        en_scores = load_scores(f"data/scores/Mesmer/ip/EN")
        ludwig_scores = load_scores(f"data/scores/Mesmer/ip/Ludwig{'_sp_' + str(spatial) if spatial > 0 else ''}")
        ae_scores = load_ae_scores(mode="ip", replace_value=replace_value, add_noise="no_noise", spatial=spatial)
        gnn_scores = load_gnn_scores(mode="ip", replace_value=replace_value, spatial=spatial)
    elif mode == 'exp':
        en_scores = load_scores(f"data/scores/Mesmer/exp/EN")
        ludwig_scores = load_scores(f"data/scores/Mesmer/exp/Ludwig{'_sp_' + str(spatial) if spatial > 0 else ''}")
        ae_scores = load_ae_scores(mode="exp", replace_value=replace_value, add_noise='no_noise', spatial=spatial)
        gnn_scores = load_gnn_scores(mode="exp", replace_value=replace_value, spatial=spatial)
    else:
        raise ValueError("Mode not recognized")

    ludwig_scores = ludwig_scores.groupby(["Marker", "Biopsy"]).head(5).reset_index()
    ludwig_scores["Network"] = "Light GBM"
    # ludwig_scores = ludwig_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()

    en_scores = en_scores.rename(columns={"Mode": "Network"})
    en_scores["Network"] = "EN"
    # duplicate rows of en socres
    en_scores = pd.concat([en_scores] * 30, ignore_index=True)

    # Select best performing iteration per marker
    ae_scores = ae_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_scores = ae_scores.groupby(["Marker", "Biopsy"]).head(5).reset_index()
    # ae_scores = ae_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    ae_scores["Network"] = f"AE"

    gnn_scores = gnn_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    gnn_scores = gnn_scores.groupby(["Marker", "Biopsy"]).head(5).reset_index()
    # gnn_scores = gnn_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    gnn_scores["Network"] = f"GNN"

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    ae_scores = ae_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    gnn_scores = gnn_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    # en_scores = en_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]

    # combine ae, light gbm and gnn scores whcih have all the same index
    scores = pd.concat([ludwig_scores, ae_scores, gnn_scores], ignore_index=True)
    # reset index
    scores = scores.reset_index(drop=True)

    create_boxen_plot(data=scores, metric=metric.upper(), title=metric.upper(), save_folder=save_path,
                      file_name=metric.upper(),
                      ylim=[0, 0.6])
