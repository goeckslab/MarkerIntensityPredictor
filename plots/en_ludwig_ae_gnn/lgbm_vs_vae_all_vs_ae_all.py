import argparse, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from statannotations.Annotator import Annotator

save_path = Path("plots/en_ludwig_ae_gnn/plots")


def load_vae_all_scores(mode: str, replace_value: str) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "vae_all", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == 0]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_ae_all_scores(mode: str, replace_value: str) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_all", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == 0]
    all_scores = all_scores[all_scores["FE"] == 0]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


def load_lgbm_scores(mode: str, spatial: int):
    all_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["FE"] == spatial]
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
                       palette={"AE ALL": "green", "VAE ALL": "red", "LGBM": "orange"})
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
    hue_order = ["LGBM", 'AE ALL', "VAE ALL"]
    pairs = [
        (("pRB", "AE ALL"), ("pRB", "VAE ALL")),
        (("CD45", "AE ALL"), ("CD45", "VAE ALL")),
        (("CK19", "AE ALL"), ("CK19", "VAE ALL")),
        (("Ki67", "AE ALL"), ("Ki67", "VAE ALL")),
        (("aSMA", "AE ALL"), ("aSMA", "VAE ALL")),
        (("Ecad", "AE ALL"), ("Ecad", "VAE ALL")),
        (("PR", "AE ALL"), ("PR", "VAE ALL")),
        (("CK14", "AE ALL"), ("CK14", "VAE ALL")),
        (("HER2", "AE ALL"), ("HER2", "VAE ALL")),
        (("AR", "AE ALL"), ("AR", "VAE ALL")),
        (("CK17", "AE ALL"), ("CK17", "VAE ALL")),
        (("p21", "AE ALL"), ("p21", "VAE ALL")),
        (("Vimentin", "AE ALL"), ("Vimentin", "VAE ALL")),
        (("pERK", "AE ALL"), ("pERK", "VAE ALL")),
        (("EGFR", "AE ALL"), ("EGFR", "VAE ALL")),
        (("ER", "AE ALL"), ("ER", "VAE ALL")),
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

    mode = str(args.mode).upper()
    metric: str = args.metric
    markers = args.markers

    save_path = Path(save_path, "lgbm_vs_ae_all_vs_vae_all")
    save_path = Path(save_path, mode)
    # save_path = Path(save_path, str(spatial))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    ae_all_scores: pd.DataFrame = load_ae_all_scores(mode=mode, replace_value="mean")
    vae_all_scores: pd.DataFrame = load_vae_all_scores(mode=mode, replace_value="mean")
    lgbm_scores: pd.DataFrame = load_lgbm_scores(mode=mode, spatial=0)

    # Select only Marker, MAE, MSE, RMSE and Biopsy
    ae_all_scores = ae_all_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    vae_all_scores = vae_all_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    lgbm_scores = lgbm_scores[["Marker", "MAE", "RMSE", "Biopsy", "Network"]]
    # combine ae and fe scores
    scores = pd.concat([lgbm_scores,ae_all_scores, vae_all_scores], axis=0)

    if markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # create violin plots

    if markers:
        file_name = f"{metric.upper()}_boxen_plot_all_markers"
    else:
        file_name = f"{metric.upper()}_boxen_plot"

    create_boxen_plot(data=scores, metric="MAE", save_folder=save_path, file_name=file_name,
                      ylim=(0, 0.8))
    # create_boxen_plot(data=scores, metric="RMSE", save_folder=save_path, file_name=file_name,
    #                  ylim=(0, 0.8))
