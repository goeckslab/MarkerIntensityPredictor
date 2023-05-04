import argparse, os
import shutil
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

save_folder = Path("plots/ae/plots")


def load_ae_scores(root_folder: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(root_folder):
        for name in files:
            if Path(name).suffix == ".csv" and name == "scores.csv":
                print("Loading file: ", name)
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, "Not all biopsies have been processed"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


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
    parser.add_argument("-r", "--radius", choices=[23, 46, 92, 138, 184], default=23, type=int)
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    args = parser.parse_args()

    radius = args.radius
    metric: str = args.metric
    mode = args.mode
    add_noise = args.an

    save_folder = Path(save_folder, "ae_zero_vs_ae_mean")
    save_folder = Path(save_folder, mode)

    if args.markers:
        save_folder = Path(save_folder, "all_markers")
    if "sp" in mode:
        save_folder = Path(save_folder, f"sp_{radius}")

    if add_noise:
        save_folder = Path(save_folder, "noise")
    else:
        save_folder = Path(save_folder, "no_noise")

    if save_folder.exists():
        shutil.rmtree(save_folder)

    print("Creating new folder")
    save_folder.mkdir(parents=True)

    if mode == "ip":
        ae_zero_path = f"ae_imputation/ip/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/ip/mean/{'noise' if add_noise else 'no_noise'}"
    elif mode == "op":
        ae_zero_path = f"ae_imputation/op/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/op/mean/{'noise' if add_noise else 'no_noise'}"
    elif mode == "exp":
        ae_zero_path = f"ae_imputation/exp/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/exp/mean/{'noise' if add_noise else 'no_noise'}"
    elif mode == "sp_23":
        ae_zero_path = f"ae_imputation/sp_23/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/sp_23/mean/{'noise' if add_noise else 'no_noise'}"
    else:
        raise ValueError("Unknown mode")

    print(ae_zero_path)

    zero_scores: pd.DataFrame = load_ae_scores(root_folder=ae_zero_path)
    zero_scores["Network"] = "AE Zero"
    mean_scores: pd.DataFrame = load_ae_scores(root_folder=ae_mean_path)
    mean_scores["Network"] = f"AE Mean"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    zero_scores = zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    zero_scores = zero_scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    mean_scores = mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    mean_scores = mean_scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    create_histogram(data=zero_scores, file_name=f"ae_zero_iteration_distribution")
    create_histogram(data=mean_scores, file_name=f"ae_mean_iteration_distribution")

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    zero_scores = zero_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]
    mean_scores = mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]

    # combine ae and fe scores
    scores = pd.concat([zero_scores, mean_scores], axis=0)

    # duplicate scores for each marker
    scores = pd.concat([scores] * 30, ignore_index=True)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if args.markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric=metric.upper(), save_folder=save_folder,
                      file_name=f"{metric.upper()}_boxen_plot",
                      ylim=(0, 1), patient_type=mode, ae_fe="sp" in mode)
