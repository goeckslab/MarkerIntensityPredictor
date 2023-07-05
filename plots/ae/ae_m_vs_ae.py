import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

save_path = Path("plots/ae/plots")


def load_ae_scores(mode: str, replace_value: str) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == 0]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_ae_m_scores(mode: str, replace_value: str) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "ae_m", "scores.csv"))
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == 0]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


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
    hue_order = ['AE Mean', "AE M Mean"]
    pairs = [
        (("pRB", "AE Mean"), ("pRB", "AE M Mean")),
        (("CD45", "AE Mean"), ("CD45", "AE M Mean")),
        (("CK19", "AE Mean"), ("CK19", "AE M Mean")),
        (("Ki67", "AE Mean"), ("Ki67", "AE M Mean")),
        (("aSMA", "AE Mean"), ("aSMA", "AE M Mean")),
        (("Ecad", "AE Mean"), ("Ecad", "AE M Mean")),
        (("PR", "AE Mean"), ("PR", "AE M Mean")),
        (("CK14", "AE Mean"), ("CK14", "AE M Mean")),
        (("HER2", "AE Mean"), ("HER2", "AE M Mean")),
        (("AR", "AE Mean"), ("AR", "AE M Mean")),
        (("CK17", "AE Mean"), ("CK17", "AE M Mean")),
        (("p21", "AE Mean"), ("p21", "AE M Mean")),
        (("Vimentin", "AE Mean"), ("Vimentin", "AE M Mean")),
        (("pERK", "AE Mean"), ("pERK", "AE M Mean")),
        (("EGFR", "AE Mean"), ("EGFR", "AE M Mean")),
        (("ER", "AE Mean"), ("ER", "AE M Mean")),
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
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("--metric", type=str, choices=["MAE", "RMSE"], default="MAE")
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

    ae_m_mean_scores: pd.DataFrame = load_ae_m_scores(mode=mode, replace_value="mean")
    ae_m_mean_scores["Network"] = "AE M Mean"
    ae_mean_scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value="mean")
    ae_mean_scores["Network"] = "AE Mean"

    # Select first iteration
    # ae_scores = ae_scores[ae_scores["Iteration"] == 0]

    # for each marker and biopsy, select only the iteration with the lowest mae
    ae_m_mean_scores = ae_m_mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_m_mean_scores = ae_m_mean_scores.groupby(["Marker", "Biopsy"]).head(5)
    ae_m_mean_scores = ae_m_mean_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    ae_m_mean_scores["Network"] = "AE Zero"
    ae_m_mean_scores["Type"] = mode

    create_histogram(data=ae_m_mean_scores, file_name=f"ae_mean_iteration_distribution", save_folder=save_path)

    ae_mean_scores = ae_mean_scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    ae_mean_scores = ae_mean_scores.groupby(["Marker", "Biopsy"]).head(5)
    ae_mean_scores = ae_mean_scores.groupby(["Marker", "Biopsy"]).mean().reset_index()
    ae_mean_scores["Network"] = "AE Mean"
    ae_mean_scores["Type"] = mode

    create_histogram(data=ae_mean_scores, file_name=f"ae_zero_iteration_distribution", save_folder=save_path)

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ae_mean_scores = ae_mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    ae_m_mean_scores = ae_m_mean_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]

    # combine ae and fe scores
    scores = pd.concat([ae_mean_scores, ae_m_mean_scores], axis=0)

    if markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_path, file_name=file_name,
                      ylim=(0, 0.8))
