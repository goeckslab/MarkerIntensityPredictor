import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

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
                      ylim: tuple, ae_fe: bool, patient_type: str):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=metric, hue="Network", split=True, cut=0)
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network")
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
    hue_order = ['AE', 'Ludwig']
    pairs = [
        (("pRB", "AE"), ("pRB", "Ludwig")),
        (("CD45", "AE"), ("CD45", "Ludwig")),
        (("CK19", "AE"), ("CK19", "Ludwig")),
        (("Ki67", "AE"), ("Ki67", "Ludwig")),
        (("aSMA", "AE"), ("aSMA", "Ludwig")),
        (("Ecad", "AE"), ("Ecad", "Ludwig")),
        (("PR", "AE"), ("PR", "Ludwig")),
        (("CK14", "AE"), ("CK14", "Ludwig")),
        (("HER2", "AE"), ("HER2", "Ludwig")),
        (("AR", "AE"), ("AR", "Ludwig")),
        (("CK17", "AE"), ("CK17", "Ludwig")),
        (("p21", "AE"), ("p21", "Ludwig")),
        (("Vimentin", "AE"), ("Vimentin", "Ludwig")),
        (("pERK", "AE"), ("pERK", "Ludwig")),
        (("EGFR", "AE"), ("EGFR", "Ludwig")),
        (("ER", "AE"), ("ER", "Ludwig")),
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
    parser.add_argument("--mode", choices=["ip", "op", "exp", "sp_23"], default="ip")
    parser.add_argument("-r", "--radius", choices=[23, 46, 92, 138, 184], default=23, type=int)
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    args = parser.parse_args()

    radius = args.radius
    mode = args.mode
    metric: str = args.metric
    add_noise = args.an

    save_folder = Path(save_folder, "ae_vs_ludwig")
    save_folder = Path(save_folder, mode)

    if args.markers:
        save_folder = Path(save_folder, "all_markers")
    if "sp" in mode:
        save_folder = Path(save_folder, f"sp_{radius}")

    if save_folder.exists():
        shutil.rmtree(save_folder)
    save_folder.mkdir(parents=True)

    if mode == "ip":
        ae_path = f"ae_imputation/ip/zero/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/in_patient/Ludwig"
    elif mode == "op":
        ae_path = f"ae_imputation/op/zero/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/out_patient/Ludwig"
    elif mode == "exp":
        ae_path = f"ae_imputation/exp/zero/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/exp/Ludwig"
    elif mode == "sp_23":
        ae_path = f"ae_imputation/sp_23/zero/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = f"data/scores/Mesmer/out_patient/Ludwig_sp_23"
    else:
        raise ValueError("Unknown mode")


    print(ae_path)
    print(ludwig_path)

    ae_scores: pd.DataFrame = load_ae_scores(root_folder=ae_path)
    ae_scores["Network"] = "AE"
    ludwig_scores: pd.DataFrame = load_fe_scores(root_folder=ludwig_path)
    ludwig_scores["Network"] = f"Ludwig"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    ae_scores = ae_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_scores = ae_scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    create_histogram(data=ae_scores, file_name=f"ae_iteration_distribution")

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]
    ae_scores = ae_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]

    # combine ae and fe scores
    scores = pd.concat([ae_scores, ludwig_scores], axis=0)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if args.markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_folder, file_name=file_name,
                      ylim=(0, 0.5), patient_type=mode, ae_fe="sp" in mode)
