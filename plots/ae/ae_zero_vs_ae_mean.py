import argparse, os
import shutil
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

save_path = Path("plots/ae/plots")


def load_ae_scores(mode: str, replace_value: str, add_noise: str, spatial: int, hyper: int):
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == hyper]
    return all_scores


def create_boxen_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str,
                      ylim: tuple, ae_fe: bool, patient_type: str):
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
    hue_order = ['AE Zero', 'AE Mean']
    pairs = [
        (("pRB", "AE Zero"), ("pRB", "AE Mean")),
        (("CD45", "AE Zero"), ("CD45", "AE Mean")),
        (("CK19", "AE Zero"), ("CK19", "AE Mean")),
        (("Ki67", "AE Zero"), ("Ki67", "AE Mean")),
        (("aSMA", "AE Zero"), ("aSMA", "AE Mean")),
        (("Ecad", "AE Zero"), ("Ecad", "AE Mean")),
        (("PR", "AE Zero"), ("PR", "AE Mean")),
        (("CK14", "AE Zero"), ("CK14", "AE Mean")),
        (("HER2", "AE Zero"), ("HER2", "AE Mean")),
        (("AR", "AE Zero"), ("AR", "AE Mean")),
        (("CK17", "AE Zero"), ("CK17", "AE Mean")),
        (("p21", "AE Zero"), ("p21", "AE Mean")),
        (("Vimentin", "AE Zero"), ("Vimentin", "AE Mean")),
        (("pERK", "AE Zero"), ("pERK", "AE Mean")),
        (("EGFR", "AE Zero"), ("EGFR", "AE Mean")),
        (("ER", "AE Zero"), ("ER", "AE Mean")),
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


def create_histogram(data: pd.DataFrame, file_name: str):
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
    parser.add_argument("--mode", choices=["ip", "op", "exp", ], default="ip")
    parser.add_argument("-sp", "--spatial", choices=[0, 23, 46, 92, 138, 184], default=0, type=int)
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    parser.add_argument("-hp", "--hyper", action="store_true", required=False, default=False,
                        help="Should hyper parameter tuning be used?")
    args = parser.parse_args()

    spatial: int = args.spatial
    metric: str = args.metric
    mode = args.mode
    add_noise: str = "noise" if args.an else "no_noise"
    markers = args.markers
    hyper = args.hyper

    save_path = Path(save_path, "ae_zero_vs_ae_mean")
    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    save_folder = Path(save_path, add_noise)

    if hyper:
        save_folder = Path(save_folder, "hyper")
    else:
        save_folder = Path(save_folder, "no_hyper")

    if save_folder.exists():
        shutil.rmtree(save_folder)

    print("Creating new folder")
    save_folder.mkdir(parents=True)

    zero_scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value="zero", add_noise=add_noise, spatial=spatial,
                                               hyper=hyper)
    zero_scores["Network"] = "AE Zero"
    mean_scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value="mean", add_noise=add_noise, spatial=spatial,
                                               hyper=hyper)
    mean_scores["Network"] = f"AE Mean"

    # select best 5 perforing iterations for each marker and each biopsy and calculate the mean
    zero_scores = zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    zero_scores = zero_scores.groupby(["Marker", "Biopsy"]).head(5)
    zero_scores = zero_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    zero_scores["Network"] = "AE Zero"
    zero_scores["Replace Value"] = "Zero"

    # select best 5 perforing iterations for each marker and each biopsy and calculate the mean
    mean_scores = mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    mean_scores = mean_scores.groupby(["Marker", "Biopsy"]).head(5)
    mean_scores = mean_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    mean_scores["Network"] = "AE Mean"
    mean_scores["Replace Value"] = "Mean"

    create_histogram(data=zero_scores, file_name=f"ae_zero_iteration_distribution")
    create_histogram(data=mean_scores, file_name=f"ae_mean_iteration_distribution")

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

    create_boxen_plot(data=scores, metric=metric.upper(), save_folder=save_folder,
                      file_name=f"{metric.upper()}_boxen_plot",
                      ylim=(0, 1), patient_type=mode, ae_fe="sp" in mode)
