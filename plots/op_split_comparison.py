import pandas as pd
from pathlib import Path
import argparse
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats

ip_vs_op_folder = Path("op_vs_ip_plots")

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]


def create_violin_plot(data: pd.DataFrame, outlier_df: pd.DataFrame, metric: str, save_folder: Path, file_name: str,
                       ylim: tuple):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    ax = sns.violinplot(data=data, x="Marker", y=metric, hue="Split", split=False, cut=0)
    #sns.stripplot(data=outlier_df, x="Marker", y=metric, jitter=True, hue="Split")

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


def load_scores() -> pd.DataFrame:
    scores = []
    for root, dirs, _ in os.walk("data/scores/Mesmer/out_patient"):
        for directory in dirs:
            if directory == "EN" or directory == "EN_6_2":
                # load only files of this folder
                for sub_root, _, files in os.walk(os.path.join(root, directory)):
                    for name in files:
                        if Path(name).suffix == ".csv":
                            print("Loading file: ", name)
                            scores.append(pd.read_csv(os.path.join(sub_root, name), sep=",", header=0))

    assert len(scores) == 16, "Not all biopsies have been loaded..."

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    args = parser.parse_args()

    scores: pd.DataFrame = load_scores()

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    if args.markers:
        mae_file_name = f"en_split_comparison_mae_violin_plot"
        rmse_file_name = f"en_split_comparison_rmse_violin_plot"
    else:
        mae_file_name = f"en_split_comparison_mae_violin_plot_all_markers"
        rmse_file_name = f"en_split_comparison_rmse_violin_plot_all_markers"

    y_lim = [0, 0.4]

    # Rename mode EN from scores dataset to 7/1 split
    scores.loc[scores["Mode"] == "EN", "Mode"] = "7/1"
    scores.loc[scores["Mode"] == "EN_6_2", "Mode"] = "6/2"
    scores.rename(columns={"Mode": "Split"}, inplace=True)

    outlier_df = []
    percent_outliers = []
    for marker in scores["Marker"].unique():
        # extract all outliers
        data = scores[scores["Marker"] == marker]

        outliers = [y for stat in boxplot_stats(data["MAE"]) for y in stat['fliers']]
        print(outliers)
        rows = data[data["MAE"].isin(outliers)]
        outlier_df.append(rows)
        # extract all rows which matches the value of the outliers

        percent_outliers.append({
            "Marker": marker,
            "Percentage": len(outliers) / len(scores) * 100,
            "Total Outliers": len(outliers),
            "Total cells": len(scores)
        })

    outlier_df = pd.concat(outlier_df, axis=0)

    print(outlier_df)

    create_violin_plot(data=scores, outlier_df=outlier_df, metric="MAE",
                       file_name=mae_file_name, save_folder=ip_vs_op_folder,
                       ylim=y_lim)
