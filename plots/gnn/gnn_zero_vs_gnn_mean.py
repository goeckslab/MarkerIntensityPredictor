import argparse, os
import shutil
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

save_path = Path("plots/gnn/plots")


def load_gnn_scores(mode: str, replace_value: str, add_noise: str, spatial: int) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "gnn", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    return all_scores


def create_boxen_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str,
                      ylim: tuple):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=metric, hue="Network", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network", palette="Set2")
    # if ae_fe:
    #     plt.title(f"AutoEncoder Zero vs AutoEncoder Mean\n Radius: {radius}")
    # else:
    #     plt.title(f"{patient_type.upper()} AutoEncoder Zero vs AutoEncoder Mean")
    # remove y axis label
    plt.ylabel(metric)
    plt.xlabel("Marker")
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
    hue_order = ['GNN Zero', 'GNN Mean']
    pairs = [
        (("pRB", "GNN Zero"), ("pRB", "GNN Mean")),
        (("CD45", "GNN Zero"), ("CD45", "GNN Mean")),
        (("CK19", "GNN Zero"), ("CK19", "GNN Mean")),
        (("Ki67", "GNN Zero"), ("Ki67", "GNN Mean")),
        (("aSMA", "GNN Zero"), ("aSMA", "GNN Mean")),
        (("Ecad", "GNN Zero"), ("Ecad", "GNN Mean")),
        (("PR", "GNN Zero"), ("PR", "GNN Mean")),
        (("CK14", "GNN Zero"), ("CK14", "GNN Mean")),
        (("HER2", "GNN Zero"), ("HER2", "GNN Mean")),
        (("AR", "GNN Zero"), ("AR", "GNN Mean")),
        (("CK17", "GNN Zero"), ("CK17", "GNN Mean")),
        (("p21", "GNN Zero"), ("p21", "GNN Mean")),
        (("Vimentin", "GNN Zero"), ("Vimentin", "GNN Mean")),
        (("pERK", "GNN Zero"), ("pERK", "GNN Mean")),
        (("EGFR", "GNN Zero"), ("EGFR", "GNN Mean")),
        (("ER", "GNN Zero"), ("ER", "GNN Mean")),
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


def create_histogram(data: pd.DataFrame, file_name: str, save_folder: Path):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    # sns.histplot(data=data, x="Iteration", hue="Marker", multiple="stack", bins=10)
    # replace _ with '' for biopsy
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values

    ax = sns.boxenplot(data=data, x="Marker", y="Iteration", dodge=True)
    # set title
    plt.title(f"Distribution of Iterations for each Marker for all Biopsies\nMode: {mode.replace('_', ' ').upper()}")
    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    ax.set_yticklabels(y_ticks, rotation=0)
    ax.set_xticklabels(x_ticks, rotation=90)

    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ip", "exp"], default="ip")
    parser.add_argument("-sp", "--spatial", choices=[23, 46, 92, 138, 184], default=23, type=int)
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
    args = parser.parse_args()

    spatial: int = args.spatial
    metric: str = args.metric
    mode = args.mode

    markers = args.markers
    save_path = Path(save_path, "gnn_zero_vs_gnn_mean")
    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print("Creating new folder..")
    save_path.mkdir(parents=True)

    zero_scores: pd.DataFrame = load_gnn_scores(mode=mode, spatial=spatial, add_noise="no_noise", replace_value="zero")
    zero_scores["Network"] = "GNN Zero"
    mean_scores: pd.DataFrame = load_gnn_scores(mode=mode, spatial=spatial, add_noise="no_noise", replace_value="mean")
    mean_scores["Network"] = f"GNN Mean"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    zero_scores = zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    zero_scores = zero_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).head(5)
    zero_scores = zero_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).mean().reset_index()
    zero_scores["Network"] = "GNN Zero"

    mean_scores = mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    mean_scores = mean_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).head(5)
    mean_scores = mean_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).mean().reset_index()
    mean_scores["Network"] = "GNN Mean"

    create_histogram(data=zero_scores, file_name=f"ae_zero_iteration_distribution", save_folder=save_path)
    create_histogram(data=mean_scores, file_name=f"ae_mean_iteration_distribution", save_folder=save_path)

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    zero_scores = zero_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    mean_scores = mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]

    # combine ae and fe scores
    scores = pd.concat([zero_scores, mean_scores], axis=0)

    if markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric=metric.upper(), save_folder=save_path,
                      file_name=f"{metric.upper()}_boxen_plot",
                      ylim=(0, 1))
