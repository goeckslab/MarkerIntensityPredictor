import argparse, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ip_source_folder = "ae/ip"
op_source_folder = "ae/op"
save_path = Path("op_vs_ip_plots/ae/")


def load_scores(hyperopt: bool) -> pd.DataFrame:
    if hyperopt:
        file_name = "hp_scores.csv"
    else:
        file_name = "scores.csv"

    scores = []
    for root, dirs, files in os.walk(ip_source_folder):
        for file in files:
            if file == file_name:
                print("Loading file: ", file)
                scores.append(pd.read_csv(os.path.join(root, file), sep=",", header=0))

    for root, dirs, files in os.walk(op_source_folder):
        for file in files:
            if file == file_name:
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
    ax = sns.violinplot(data=data, x="Marker", y=score, hue="Mode", split=True, cut=0)

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
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
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    parser.add_argument("-hp", "--hyperopt", action="store_true", default=False)
    args = parser.parse_args()

    metric: str = args.scores
    hyperopt: bool = args.hyperopt

    if hyperopt:
        print("Using hyperopt")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    scores = load_scores(hyperopt=hyperopt)
    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    if args.markers:
        create_violin_plot(data=scores, score=metric, save_folder=save_path,
                           file_name=f"{metric.lower()}_ae_denoising_violin{'_hyper' if hyperopt else ''}",
                           ylim=(0, 0.5))
    else:
        create_violin_plot(data=scores, score=metric, save_folder=save_path,
                           file_name=f"{metric.lower()}_ae_denoising_violin_all_markers{'_hyper' if hyperopt else ''}",
                           ylim=(0, 0.5))
