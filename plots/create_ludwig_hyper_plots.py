import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path
import argparse
from typing import List

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]

ip_folder = Path("ip_plots")
op_folder = Path("op_plots")
ip_vs_op_folder = Path("op_vs_ip_plots")


def truncate_decimals(target_allocation, two_decimal_places) -> float:
    decimal_exponent = 10.0 ** two_decimal_places
    return math.trunc(decimal_exponent * target_allocation) / decimal_exponent


def rule(row, column, decimal_precision=4):
    number = str(row[column])
    number = float(number.rstrip("%"))

    if number > math.floor(number) + 0.5:
        number = round(number, decimal_precision)
        # print("This value is being rounded", number)

    elif number < math.ceil(number) - 0.5:
        number = truncate_decimals(number, decimal_precision)
        # print("This value is being truncated", number)

    # else:
    # print("This value does not meet one of the above conditions", round(number, decimal_precision))

    return number


def create_heatmap_df(data, column, decimal_precision=4):
    heatmap_df = data.copy()
    heatmap_df["Biopsy"] = [biopsy.replace('_', ' ') for biopsy in list(heatmap_df["Biopsy"].values)]
    heatmap_df[column] = heatmap_df.apply(rule, axis=1, args=(column, decimal_precision,))

    return heatmap_df


def plot_performance_heatmap_per_segmentation(data, score: str, segmentation: str, folder: Path, file_name: str):
    data = create_heatmap_df(data, score, decimal_precision=4)

    data = data.pivot(index="Biopsy", columns="Marker", values=score)
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]

    save_folder = Path(f"{folder}/ludwig_hyper")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=0, vmax=0.5, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.title(f"{'In' if 'ip_' in str(folder) else 'Out'} Patient {score} Scores \n {segmentation}")
    plt.xlabel("Biopsy")
    plt.ylabel(score)
    plt.tight_layout()
    plt.savefig(Path(f"{save_folder}/{file_name}.png"))
    plt.close('all')


def create_violin_plot_per_segmentation(data: pd.DataFrame, score: str, title: str, save_folder: Path, file_name: str,
                                        ylim: List):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    ax = sns.violinplot(data=data, x="Marker", y=score, hue="Type", split=False, cut=0)

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


def load_scores() -> pd.DataFrame:
    scores = []
    for root, dirs, _ in os.walk("data/scores/Mesmer"):
        for directory in dirs:
            if directory == "Ludwig_Hyper":
                # load only files of this folder
                for sub_root, _, files in os.walk(os.path.join(root, directory)):
                    for name in files:
                        if Path(name).suffix == ".csv":
                            print("Loading file: ", name)
                            scores.append(pd.read_csv(os.path.join(sub_root, name), sep=",", header=0))

    assert len(scores) == 16, "Not all biopsies have been processed"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':
    # argsparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    args = parser.parse_args()

    # load mesmer mae scores from data mesmer folder and all subfolders

    scores: pd.DataFrame = load_scores()

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # Create bar plot which compares in patient performance of the different segementations for each biopsy
    # The bar plot should be saved in the plots folder

    ip_mae_scores = scores[scores["Type"] == "IP"].copy()

    mae_performance_data = ip_mae_scores.copy()
    mae_performance_data.drop(columns=["Type", "FE", "Mode", "Hyper", "Panel"], inplace=True)

    # Plot mesmer
    mae_performance_data_mesmer = mae_performance_data[mae_performance_data["Segmentation"] == "Mesmer"].copy()
    mae_performance_data_mesmer.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MAE", "Mesmer", folder=ip_folder,
                                              file_name="ludwig_hyper_ip_mesmer_mae_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MSE", "Mesmer", folder=ip_folder,
                                              file_name="ludwig_hyper_ip_mesmer_mse_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "RMSE", "Mesmer", folder=ip_folder,
                                              file_name=
                                              "ludwig_hyper_ip_mesmer_rmse_heatmap")

    # Out Patient

    op_mae_scores = scores[scores["Type"] == "OP"].copy()

    mae_performance_data = op_mae_scores.copy()
    mae_performance_data.drop(columns=["Type", "FE", "Mode", "Hyper", "Panel"], inplace=True)

    # Plot mesmer
    mae_performance_data_mesmer = mae_performance_data[mae_performance_data["Segmentation"] == "Mesmer"].copy()
    mae_performance_data_mesmer.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MAE", "Mesmer", folder=op_folder,
                                              file_name="ludwig_hyper_op_mesmer_mae_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MSE", "Mesmer", folder=op_folder,
                                              file_name="ludwig_hyper_op_mesmer_mse_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "RMSE", "Mesmer", folder=op_folder,
                                              file_name=
                                              "ludwig_hyper_op_mesmer_rmse_heatmap")

    # Violin plots for out & in patient data for each segmentation

    data = pd.concat([ip_mae_scores, op_mae_scores])

    print(data["Segmentation"].unique())
    for segmentation in data["Segmentation"].unique():
        data_seg = data[data["Segmentation"] == segmentation].copy()
        data_seg.drop(columns=["SNR", "Segmentation"], inplace=True)
        data_seg["Biopsy"] = data_seg["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values

        y_lim = [0, 0.4]

        if args.markers:
            mae_file_name = f"ludwig_hyper_{segmentation}_mae_violin_plot"
            rmse_file_name = f"ludwig_hyper_{segmentation}_rmse_violin_plot"
        else:
            mae_file_name = f"ludwig_hyper_{segmentation}_mae_violin_plot_all_markers"
            rmse_file_name = f"ludwig_hyper_{segmentation}_rmse_violin_plot_all_markers"

        create_violin_plot_per_segmentation(data=data_seg, score="MAE",
                                            title=f"In & Out patient performance using Ludwig + Hyperopt for {segmentation}",
                                            file_name=mae_file_name, save_folder=ip_vs_op_folder,
                                            ylim=y_lim)

        create_violin_plot_per_segmentation(data=data_seg, score="RMSE",
                                            title=f"In & Out patient performance using Ludwig + Hyperopt for {segmentation}",
                                            file_name=rmse_file_name,
                                            save_folder=ip_vs_op_folder, ylim=y_lim)