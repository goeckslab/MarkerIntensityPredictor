import argparse, os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List

save_folder = Path("ip_plots")

if not save_folder.exists():
    save_folder.mkdir(parents=True, exist_ok=True)


def truncate_decimals(target_allocation, two_decimal_places) -> float:
    decimal_exponent = 10.0 ** two_decimal_places
    return math.trunc(decimal_exponent * target_allocation) / decimal_exponent


def rule(row, decimal_precision=4):
    number = str(row.Score)
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


def plot_difference_heatmap(data, columns: List, suptitle: str, title: str, file_name: str):
    data.drop(columns=columns,
              inplace=True)
    data.rename(columns={"Difference": "Score"}, inplace=True)
    data = create_heatmap_df(data, decimal_precision=3)
    data = data.pivot(index="Biopsy", columns="Marker", values="Score")
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=-0.2, vmax=0.3, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.xlabel('Feature')
    plt.suptitle(suptitle, x=0.45, fontsize=16)
    plt.title(title, x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")


def plot_difference_line_plot(data: pd.DataFrame, title: str, file_name: str):
    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.lineplot(data=data, x="Marker", y="Score", hue="Biopsy", style="Segmentation")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def plot_difference_violin_plot(data: pd.DataFrame, title: str, file_name: str):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.violinplot(data=data, x="Marker", y="Score", hue="Segmentation")
    plt.title(title)
    plt.legend(loc='upper center')
    plt.ylim(-0.3, 0.9)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_clustermap(data: pd.DataFrame, title: str, file_name: str):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    data.sort_values(by=["Biopsy", "Marker"], inplace=True)
    data = data.pivot(index="Marker", columns="Biopsy", values="Score")

    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.clustermap(data=data, vmin=0, vmax=0.5)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_heatmap_df(data, decimal_precision=4):
    heatmap_df = data.copy()
    heatmap_df["Biopsy"] = [biopsy.replace('_', ' ') for biopsy in list(heatmap_df["Biopsy"].values)]
    heatmap_df["Score"] = heatmap_df.apply(rule, axis=1, args=(decimal_precision,))

    return heatmap_df


if __name__ == '__main__':
    biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
    # load mesmer mae scores from data mesmer folder and all subfolders

    mae_scores = []
    for root, dirs, files in os.walk("mesmer"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_non_af"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_af"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(mae_scores) == 48, "There should be 48 mae scores files but only {} were found".format(len(mae_scores))
    mae_scores = pd.concat(mae_scores, axis=0).reset_index(drop=True)

    ip = mae_scores[mae_scores["Type"] == "IP"].copy()
    ip.drop(columns=["Type", "Panel"], inplace=True)

    data = ip.copy()
    # create new columns data origin, which indicates whether the data is from mesmer or unmicst and af corrected or not
    data["Origin"] = data.apply(lambda x: "Mesmer" if "Mesmer" in x["Segmentation"] else "UnMICST + S3 AF Corrected",
                                axis=1)
    # rename origin to unmicst + s3 Non AF Corrected if not af corrected
    data.loc[data["AF Corrected"] == 0, "Origin"] = "UnMICST + S3 Non AF Corrected"

    plt.figure(figsize=(20, 10))
    sns.barplot(data=data, x="Marker", y="Score", hue="Origin", errorbar="sd")
    plt.title("MAE scores for IP biopsies\nComparison of different segmentation methods")
    plt.savefig(f"{save_folder}/ip_mae_scores.png")

    # Create heatmap for uncorrected unmicst  + s3
    data = ip[(ip["Segmentation"] == "Unmicst + S3") & (ip["AF Corrected"] == 0)].copy().reset_index(drop=True)
    data.drop(columns=["Segmentation"], inplace=True)
    data = create_heatmap_df(data)
    data = data.pivot(index="Biopsy", columns="Marker", values="Score")
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=0, vmax=0.5, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.xlabel('Feature')
    plt.suptitle("In Patient MAE Scores\nUnMICST + S3 Segmenter\nNon AF corrected", x=0.45, fontsize=16)
    plt.title("Lower values indicate lower errors", x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/ip_unmicst_s3_non_af_mae_scores_heatmap.png")

    # Create heatmap for corrected unmicst  + s3
    data = ip[(ip["Segmentation"] == "Unmicst + S3") & (ip["AF Corrected"] == 1)].copy().reset_index(drop=True)
    data.drop(columns=["Segmentation"], inplace=True)
    data = create_heatmap_df(data)
    data = data.pivot(index="Biopsy", columns="Marker", values="Score")
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=0, vmax=0.5, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.xlabel('Feature')
    plt.suptitle("In Patient MAE Scores\nUnMICST + S3 Segmenter\n AF corrected", x=0.45, fontsize=16)
    plt.title("Lower values indicate lower errors", x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/ip_unmicst_s3_af_mae_scores_heatmap.png")

    data = ip[ip["Segmentation"] == "Mesmer"].copy().reset_index(drop=True)
    data.drop(columns=["Segmentation"], inplace=True)
    data = create_heatmap_df(data)
    data = data.pivot(index="Biopsy", columns="Marker", values="Score")
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=0, vmax=0.5, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.xlabel('Feature')
    plt.suptitle("In Patient MAE Scores\nMesmer", x=0.45, fontsize=16)
    plt.title("Lower values indicate lower errors", x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/ip_mesmer_mae_scores_heatmap.png")

    # calculate difference between mesmer and unmicst separation
    s3_af = ip[(ip["Segmentation"] == "Unmicst + S3") & (ip["AF Corrected"] == 1)].copy().reset_index(drop=True)
    s3_non_af = ip[(ip["Segmentation"] == "Unmicst + S3") & (ip["AF Corrected"] == 0)].copy().reset_index(drop=True)
    mesmer = ip[ip["Segmentation"] == "Mesmer"].copy().reset_index(drop=True)

    # merge s3_af sf3_non_af
    s3_af_vs_s3_non_af = pd.merge(s3_non_af, s3_af, on=["Biopsy", "Marker"], suffixes=("_s3_non_af", "_s3_af"))
    s3_af_vs_s3_non_af["Difference"] = s3_af_vs_s3_non_af["Score_s3_non_af"].values - s3_af_vs_s3_non_af[
        "Score_s3_af"].values

    # merge s3_af and mesmer
    s3_af_vs_mesmer = pd.merge(s3_af, mesmer, on=["Biopsy", "Marker"], suffixes=("_s3_af", "_mesmer"))
    # calculate difference between s3 and mesmer
    s3_af_vs_mesmer["Difference"] = s3_af_vs_mesmer["Score_s3_af"].values - s3_af_vs_mesmer["Score_mesmer"].values

    # merge s3_non_af and mesmer
    s3_non_af_vs_mesmer = pd.merge(s3_non_af, mesmer, on=["Biopsy", "Marker"], suffixes=("_s3_non_af", "_mesmer"))
    # calculate difference between s3 and mesmer
    s3_non_af_vs_mesmer["Difference"] = s3_non_af_vs_mesmer["Score_s3_non_af"].values - s3_non_af_vs_mesmer[
        "Score_mesmer"].values

    data = s3_af_vs_mesmer.copy()
    plot_difference_heatmap(data=data,
                            columns=["Segmentation_s3_af", "Score_s3_af", "Score_mesmer", "Segmentation_mesmer"],
                            suptitle="In Patient Performance Difference\nMesmer vs. UnMicst + S3 (AF corrected)",
                            title="Negative values indicate better performance using UnMICST + S3\nPositive values indicate better performance using Mesmer",
                            file_name="ip_difference_mesmer_s3_af")

    data = s3_non_af_vs_mesmer.copy()
    plot_difference_heatmap(data=data,
                            columns=["Segmentation_s3_non_af", "Score_s3_non_af", "Score_mesmer",
                                     "Segmentation_mesmer"],
                            suptitle="In Patient Performance Difference\nMesmer vs. UnMicst + S3 (Non AF corrected)",
                            title="Negative values indicate better performance using UnMICST + S3\nPositive values indicate better performance using Mesmer",
                            file_name="ip_difference_mesmer_s3_non_af")

    data = s3_af_vs_s3_non_af.copy()
    plot_difference_heatmap(data=data,
                            columns=["Segmentation_s3_non_af", "Score_s3_non_af", "Segmentation_s3_af", "Score_s3_af"],
                            suptitle="In Patient Performance Difference\nUnMicst + S3 (AF corrected) vs. UnMicst + S3 (Non AF corrected)",
                            title="Negative values indicate better performance using UnMICST + S3 (Non AF corrected)\nPositive values indicate better performance using UnMicst + S3 (AF corrected)",
                            file_name="ip_s3_af_vs_s3_non_af")

    #  Select s3 + unmicst af corrected and mesmer
    data = ip.copy()
    data.sort_values(by=["Biopsy", "Marker"], inplace=True)
    data = data[data["AF Corrected"] == 1]

    # rename UnMICST + S3 to UnMICST + S3 (AF corrected)
    data.loc[data["Segmentation"] == "Unmicst + S3", "Segmentation"] = "Unmicst + S3 (AF corrected)"
    plot_difference_violin_plot(data=data,
                                title="In Patient Performance Difference\nSpread between biopsies\nMesmer vs. UnMicst + S3 (AF corrected)",
                                file_name="ip_difference_mesmer_s3_af_violin_plot")
    plot_difference_line_plot(data=data,
                              title="In Patient Performance Difference\nMesmer vs. UnMicst + S3 (AF corrected)",
                              file_name="ip_difference_mesmer_vs_s3_non_af_line_plot")

    # Select s3 + unmicst non af corrected and mesmer
    data = ip.copy()
    data.sort_values(by=["Biopsy", "Marker"], inplace=True)
    data = data[
        (data["Segmentation"] == "Unmicst + S3") & (data["AF Corrected"] == 0) | (data["Segmentation"] == "Mesmer")]

    # rename UnMICST + S3 to UnMICST + S3 (AF corrected)
    data.loc[data["Segmentation"] == "Unmicst + S3", "Segmentation"] = "Unmicst + S3 (Non AF corrected)"
    plot_difference_violin_plot(data=data,
                                title="In Patient Performance Difference\nSpread between biopsies\nMesmer vs. UnMicst + S3 (Non AF corrected)",
                                file_name="ip_difference_mesmer_s3_non_af_violin_plot")
    plot_difference_line_plot(data=data,
                              title="In Patient Performance Difference\nMesmer vs. UnMicst + S3 (Non AF corrected)",
                              file_name="ip_difference_mesmer_s3_non_af_line_plot")

    # s3_af clustermap
    data = s3_af.copy()
    create_clustermap(data=data, title="In Patient UnMICST + S3 Segmenter (AF corrected) \nBiopsy vs. Feature",
                      file_name="ip_cluster_map_unmicst_s3_af")

    # s3_non_af clustermap
    data = s3_non_af.copy()
    create_clustermap(data=data, title="In Patient UnMICST + S3 Segmenter (Non AF corrected) \nBiopsy vs. Feature",
                      file_name="ip_cluster_map_unmicst_s3_non_af")

    # mesmer clustermap
    data = mesmer.copy()
    create_clustermap(data=data, title="In Patient Mesmer Segmenter \nBiopsy vs. Feature",
                      file_name="ip_cluster_map_mesmer")
