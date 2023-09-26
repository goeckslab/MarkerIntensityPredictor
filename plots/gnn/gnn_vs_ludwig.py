import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

save_path = Path("plots/gnn/plots")


def load_gnn_scores(mode: str, replace_value: str, add_noise: str, spatial: int) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "gnn", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    return all_scores


def load_fe_scores(root_folder: str):
    scores = []
    for root, dirs, files in os.walk(root_folder):
        for name in files:
            if Path(name).suffix == ".csv":
                print("Loading file: ", name)
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, "Not all biopsies have been processed"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


def create_boxen_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str,
                      ylim: tuple):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=metric, hue="Network", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network",
                       palette={"GNN Mean": "orange", "GNN Zero": "blue", "Light GBM": "green"})
    # if ae_fe:
    #    plt.title(f"FE AutoEncoder vs FE Ludwig\n Radius: {radius}")
    # else:
    #    plt.title(f"{patient_type.upper()} AutoEncoder vs Ludwig")
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
    hue_order = ['GNN Mean', 'GNN Zero', 'Light GBM']
    pairs = [
        (("pRB", "GNN Mean"), ("pRB", "Light GBM")),
        (("CD45", "GNN Mean"), ("CD45", "Light GBM")),
        (("CK19", "GNN Mean"), ("CK19", "Light GBM")),
        (("Ki67", "GNN Mean"), ("Ki67", "Light GBM")),
        (("aSMA", "GNN Mean"), ("aSMA", "Light GBM")),
        (("Ecad", "GNN Mean"), ("Ecad", "Light GBM")),
        (("PR", "GNN Mean"), ("PR", "Light GBM")),
        (("CK14", "GNN Mean"), ("CK14", "Light GBM")),
        (("HER2", "GNN Mean"), ("HER2", "Light GBM")),
        (("AR", "GNN Mean"), ("AR", "Light GBM")),
        (("CK17", "GNN Mean"), ("CK17", "Light GBM")),
        (("p21", "GNN Mean"), ("p21", "Light GBM")),
        (("Vimentin", "GNN Mean"), ("Vimentin", "Light GBM")),
        (("pERK", "GNN Mean"), ("pERK", "Light GBM")),
        (("EGFR", "GNN Mean"), ("EGFR", "Light GBM")),
        (("ER", "GNN Mean"), ("ER", "Light GBM")),
        (("pRB", "GNN Zero"), ("pRB", "Light GBM")),
        (("CD45", "GNN Zero"), ("CD45", "Light GBM")),
        (("CK19", "GNN Zero"), ("CK19", "Light GBM")),
        (("Ki67", "GNN Zero"), ("Ki67", "Light GBM")),
        (("aSMA", "GNN Zero"), ("aSMA", "Light GBM")),
        (("Ecad", "GNN Zero"), ("Ecad", "Light GBM")),
        (("PR", "GNN Zero"), ("PR", "Light GBM")),
        (("CK14", "GNN Zero"), ("CK14", "Light GBM")),
        (("HER2", "GNN Zero"), ("HER2", "Light GBM")),
        (("AR", "GNN Zero"), ("AR", "Light GBM")),
        (("CK17", "GNN Zero"), ("CK17", "Light GBM")),
        (("p21", "GNN Zero"), ("p21", "Light GBM")),
        (("Vimentin", "GNN Zero"), ("Vimentin", "Light GBM")),
        (("pERK", "GNN Zero"), ("pERK", "Light GBM")),
        (("EGFR", "GNN Zero"), ("EGFR", "Light GBM")),
        (("ER", "GNN Zero"), ("ER", "Light GBM")),
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

    mode = args.mode
    metric: str = args.metric
    markers = args.markers
    spatial: int = args.spatial

    save_path = Path(save_path, "gnn_vs_ludwig")
    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")


    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    if mode == "ip":
        ludwig_path = f"data/scores/Mesmer/ip/Ludwig_sp_{spatial}"
    elif mode == "exp":
        ludwig_path = f"data/scores/Mesmer/exp/Ludwig_sp_{spatial}"
    else:
        raise ValueError("Unknown mode")

    gnn_zero_scores: pd.DataFrame = load_gnn_scores(mode=mode, replace_value="zero",
                                                    add_noise="no_noise", spatial=spatial)
    gnn_zero_scores["Network"] = "GNN Zero"
    gnn_mean_scores: pd.DataFrame = load_gnn_scores(mode=mode, replace_value="mean",
                                                    add_noise="no_noise", spatial=spatial)
    gnn_mean_scores["Network"] = "GNN Mean"

    ludwig_scores: pd.DataFrame = load_fe_scores(root_folder=ludwig_path)
    ludwig_scores["Network"] = f"Light GBM"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]


    # for each marker and biopsy, select only the iteration with the lowest mae
    gnn_zero_scores = gnn_zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    gnn_zero_scores = gnn_zero_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).head(5)
    gnn_zero_scores = gnn_zero_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).mean().reset_index()
    gnn_zero_scores["Network"] = "GNN Zero"
    create_histogram(data=gnn_zero_scores, file_name=f"gnn_mean_iteration_distribution", save_folder=save_path)

    gnn_mean_scores = gnn_mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    gnn_mean_scores = gnn_mean_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).head(5)
    gnn_mean_scores = gnn_mean_scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).mean().reset_index()
    gnn_mean_scores["Network"] = "GNN Mean"
    create_histogram(data=gnn_mean_scores, file_name=f"gnn_zero_iteration_distribution", save_folder=save_path)

    # if column combination exists in ludwig scores ranem to type
    if "Combination" in ludwig_scores.columns:
        ludwig_scores = ludwig_scores.rename(columns={"Combination": "Type"})

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    gnn_mean_scores = gnn_mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    gnn_zero_scores = gnn_zero_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]

    # combine gnn and fe scores
    scores = pd.concat([gnn_mean_scores, gnn_zero_scores, ludwig_scores], axis=0)

    if markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_path, file_name=file_name,
                      ylim=(0, 0.8))
