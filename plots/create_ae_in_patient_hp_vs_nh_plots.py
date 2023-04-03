import argparse, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

source_folder = "ae/ip"
save_path = Path("ip_plots/ae/")


def load_scores() -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file == "scores.csv" or file == "hp_scores.csv":
                print("Loading file: ", file)
                scores.append(pd.read_csv(os.path.join(root, file), sep=",", header=0))

    assert len(scores) == 16, f"Not all biopsies have been selected. Only {len(scores)} biopsies have been selected."

    return pd.concat(scores, axis=0).sort_values(by=["Marker"]).reset_index(drop=True)


def create_violin_plot(data: pd.DataFrame, score: str, save_folder: Path, file_name: str,
                       ylim: tuple):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    ax = sns.violinplot(data=data, x="Marker", y=score, hue="HP", split=True, cut=0)

    # plt.title(title)
    # remove y axis label
    # plt.ylabel("")
    # plt.xlabel("")
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
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_line_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax = sns.lineplot(x="Marker", y=metric, hue="HP", data=data)
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    args = parser.parse_args()

    metric: str = args.scores

    if not save_path.exists():
        save_path.mkdir(parents=True)

    scores = load_scores()
    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    if args.markers:
        create_violin_plot(data=scores, score=metric, save_folder=save_path,
                           file_name=f"denoising_{metric.lower()}_ae_hp_vs_nh_violin",
                           ylim=(0, 0.5))
    else:
        create_violin_plot(data=scores, score=metric, save_folder=save_path,
                           file_name=f"denoising_{metric.lower()}_ae_hp_vs_nh_violin_all_markers", ylim=(0, 0.5))

    if args.markers:
        create_line_plot(data=scores, metric=metric.upper(), save_folder=save_path,
                         file_name=f"denoising_{metric.lower()}_ae_hp_vs_nh_line")
    else:
        create_line_plot(data=scores, metric=metric.upper(), save_folder=save_path,
                         file_name=f"denoising_{metric.lower()}_ae_hp_vs_nh_line_all_markers")
