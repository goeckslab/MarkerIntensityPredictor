import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

save_path = Path("plots/ae/plots")


def load_ae_scores(mode: str, replace_value: str, add_noise: str, spatial: int) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
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
                       palette={"AE Mean": "orange", "AE Zero": "blue", "Light GBM": "green"})
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
    hue_order = ['AE Mean', "AE Zero", 'Light GBM']
    pairs = [
        (("pRB", "AE Mean"), ("pRB", "Light GBM")),
        (("CD45", "AE Mean"), ("CD45", "Light GBM")),
        (("CK19", "AE Mean"), ("CK19", "Light GBM")),
        (("Ki67", "AE Mean"), ("Ki67", "Light GBM")),
        (("aSMA", "AE Mean"), ("aSMA", "Light GBM")),
        (("Ecad", "AE Mean"), ("Ecad", "Light GBM")),
        (("PR", "AE Mean"), ("PR", "Light GBM")),
        (("CK14", "AE Mean"), ("CK14", "Light GBM")),
        (("HER2", "AE Mean"), ("HER2", "Light GBM")),
        (("AR", "AE Mean"), ("AR", "Light GBM")),
        (("CK17", "AE Mean"), ("CK17", "Light GBM")),
        (("p21", "AE Mean"), ("p21", "Light GBM")),
        (("Vimentin", "AE Mean"), ("Vimentin", "Light GBM")),
        (("pERK", "AE Mean"), ("pERK", "Light GBM")),
        (("EGFR", "AE Mean"), ("EGFR", "Light GBM")),
        (("ER", "AE Mean"), ("ER", "Light GBM")),
        (("pRB", "AE Zero"), ("pRB", "Light GBM")),
        (("CD45", "AE Zero"), ("CD45", "Light GBM")),
        (("CK19", "AE Zero"), ("CK19", "Light GBM")),
        (("Ki67", "AE Zero"), ("Ki67", "Light GBM")),
        (("aSMA", "AE Zero"), ("aSMA", "Light GBM")),
        (("Ecad", "AE Zero"), ("Ecad", "Light GBM")),
        (("PR", "AE Zero"), ("PR", "Light GBM")),
        (("CK14", "AE Zero"), ("CK14", "Light GBM")),
        (("HER2", "AE Zero"), ("HER2", "Light GBM")),
        (("AR", "AE Zero"), ("AR", "Light GBM")),
        (("CK17", "AE Zero"), ("CK17", "Light GBM")),
        (("p21", "AE Zero"), ("p21", "Light GBM")),
        (("Vimentin", "AE Zero"), ("Vimentin", "Light GBM")),
        (("pERK", "AE Zero"), ("pERK", "Light GBM")),
        (("EGFR", "AE Zero"), ("EGFR", "Light GBM")),
        (("ER", "AE Zero"), ("ER", "Light GBM")),
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
    parser.add_argument("-sp", "--spatial", choices=[0, 23, 46, 92, 138, 184], default=0, type=int)
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    args = parser.parse_args()

    mode = args.mode
    metric: str = args.metric
    add_noise: str = "noise" if args.an else "no_noise"
    markers = args.markers
    spatial: int = args.spatial

    save_path = Path(save_path, "ae_vs_ludwig")
    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    save_path = Path(save_path, add_noise)

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    if mode == "ip":
        ludwig_path = "data/scores/Mesmer/ip/Ludwig" if spatial == 0 else f"data/scores/Mesmer/ip/Ludwig_sp_{spatial}"
    elif mode == "exp":
        ludwig_path = "data/scores/Mesmer/exp/Ludwig"
    else:
        raise ValueError("Unknown mode")

    ae_zero_scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value="zero",
                                                  add_noise=add_noise, spatial=spatial)
    ae_zero_scores["Network"] = "AE Zero"
    ae_mean_scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value="mean",
                                                  add_noise=add_noise, spatial=spatial)
    ae_mean_scores["Network"] = "AE Mean"

    ludwig_scores: pd.DataFrame = load_fe_scores(root_folder=ludwig_path)
    ludwig_scores["Network"] = f"Light GBM"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    ae_zero_scores = ae_zero_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_zero_scores = ae_zero_scores.groupby(["Marker", "Biopsy"]).head(5)
    ae_zero_scores = ae_zero_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    ae_zero_scores["Network"] = "AE Zero"
    ae_zero_scores["Type"] = mode

    create_histogram(data=ae_zero_scores, file_name=f"ae_mean_iteration_distribution", save_folder=save_path)

    ae_mean_scores = ae_mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_mean_scores = ae_mean_scores.groupby(["Marker", "Biopsy"]).head(5)
    ae_mean_scores = ae_mean_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    ae_mean_scores["Network"] = "AE Mean"
    ae_mean_scores["Type"] = mode

    create_histogram(data=ae_mean_scores, file_name=f"ae_zero_iteration_distribution", save_folder=save_path)

    if "Combination" in ludwig_scores.columns:
        # rename Combination to Type
        ludwig_scores = ludwig_scores.rename(columns={"Combination": "Type"})

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ludwig_scores = ludwig_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    ae_mean_scores = ae_mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    ae_zero_scores = ae_zero_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]

    # combine ae and fe scores
    scores = pd.concat([ae_mean_scores, ae_zero_scores, ludwig_scores], axis=0)

    if markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_path, file_name=file_name,
                      ylim=(0, 0.8))
