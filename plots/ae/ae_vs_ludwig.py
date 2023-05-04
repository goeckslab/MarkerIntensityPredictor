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
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network",
                       palette={"AE Mean": "orange", "AE Zero": "blue", "Ludwig": "green"})
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
    hue_order = ['AE Mean', "AE Zero", 'Ludwig']
    pairs = [
        (("pRB", "AE Mean"), ("pRB", "Ludwig")),
        (("CD45", "AE Mean"), ("CD45", "Ludwig")),
        (("CK19", "AE Mean"), ("CK19", "Ludwig")),
        (("Ki67", "AE Mean"), ("Ki67", "Ludwig")),
        (("aSMA", "AE Mean"), ("aSMA", "Ludwig")),
        (("Ecad", "AE Mean"), ("Ecad", "Ludwig")),
        (("PR", "AE Mean"), ("PR", "Ludwig")),
        (("CK14", "AE Mean"), ("CK14", "Ludwig")),
        (("HER2", "AE Mean"), ("HER2", "Ludwig")),
        (("AR", "AE Mean"), ("AR", "Ludwig")),
        (("CK17", "AE Mean"), ("CK17", "Ludwig")),
        (("p21", "AE Mean"), ("p21", "Ludwig")),
        (("Vimentin", "AE Mean"), ("Vimentin", "Ludwig")),
        (("pERK", "AE Mean"), ("pERK", "Ludwig")),
        (("EGFR", "AE Mean"), ("EGFR", "Ludwig")),
        (("ER", "AE Mean"), ("ER", "Ludwig")),
        (("pRB", "AE Zero"), ("pRB", "Ludwig")),
        (("CD45", "AE Zero"), ("CD45", "Ludwig")),
        (("CK19", "AE Zero"), ("CK19", "Ludwig")),
        (("Ki67", "AE Zero"), ("Ki67", "Ludwig")),
        (("aSMA", "AE Zero"), ("aSMA", "Ludwig")),
        (("Ecad", "AE Zero"), ("Ecad", "Ludwig")),
        (("PR", "AE Zero"), ("PR", "Ludwig")),
        (("CK14", "AE Zero"), ("CK14", "Ludwig")),
        (("HER2", "AE Zero"), ("HER2", "Ludwig")),
        (("AR", "AE Zero"), ("AR", "Ludwig")),
        (("CK17", "AE Zero"), ("CK17", "Ludwig")),
        (("p21", "AE Zero"), ("p21", "Ludwig")),
        (("Vimentin", "AE Zero"), ("Vimentin", "Ludwig")),
        (("pERK", "AE Zero"), ("pERK", "Ludwig")),
        (("EGFR", "AE Zero"), ("EGFR", "Ludwig")),
        (("ER", "AE Zero"), ("ER", "Ludwig")),
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
        ae_zero_path = f"ae_imputation/ip/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/ip/mean/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/in_patient/Ludwig"
    elif mode == "op":
        ae_zero_path = f"ae_imputation/op/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/op/mean/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/out_patient/Ludwig"
    elif mode == "exp":
        ae_zero_path = f"ae_imputation/exp/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/exp/mean/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = "data/scores/Mesmer/exp/Ludwig"
    elif mode == "sp_23":
        ae_zero_path = f"ae_imputation/sp_23/zero/{'noise' if add_noise else 'no_noise'}"
        ae_mean_path = f"ae_imputation/sp_23/mean/{'noise' if add_noise else 'no_noise'}"
        ludwig_path = f"data/scores/Mesmer/out_patient/Ludwig_sp_23_6_2"
    else:
        raise ValueError("Unknown mode")

    ae_mean_scores: pd.DataFrame = load_ae_scores(root_folder=ae_mean_path)
    ae_mean_scores["Network"] = "AE Mean"

    ae_zero_scores: pd.DataFrame = load_ae_scores(root_folder=ae_zero_path)
    ae_zero_scores["Network"] = "AE Zero"

    ludwig_scores: pd.DataFrame = load_fe_scores(root_folder=ludwig_path)
    ludwig_scores["Network"] = f"Ludwig"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    ae_zero_scores = ae_zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_zero_scores = ae_zero_scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    create_histogram(data=ae_zero_scores, file_name=f"ae_mean_iteration_distribution")

    ae_mean_scores = ae_mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_mean_scores = ae_mean_scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    create_histogram(data=ae_mean_scores, file_name=f"ae_zero_iteration_distribution")

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]
    ae_mean_scores = ae_mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]
    ae_zero_scores = ae_zero_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network", "Type"]]

    # combine ae and fe scores
    scores = pd.concat([ae_mean_scores, ae_zero_scores, ludwig_scores], axis=0)
    scores = pd.concat([scores] * 30, ignore_index=True)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if args.markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_folder, file_name=file_name,
                      ylim=(0, 0.8), patient_type=mode, ae_fe="sp" in mode)
