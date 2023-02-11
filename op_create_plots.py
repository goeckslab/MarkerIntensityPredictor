import argparse, os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

save_folder = Path("op_plots")

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


def create_heatmap_df(data):
    heatmap_df = data.copy()
    heatmap_df["Biopsy"] = [biopsy.replace('_', ' ') for biopsy in list(heatmap_df["Biopsy"].values)]
    heatmap_df["Score"] = heatmap_df.apply(rule, axis=1)

    return heatmap_df


if __name__ == '__main__':
    biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
    # load mesmer mae scores from data mesmer folder and all subfolders

    mae_scores = []
    for root, dirs, files in os.walk("mesmer"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(mae_scores) == 32, "There should be 32 mae scores files"
    mae_scores = pd.concat(mae_scores, axis=0).reset_index(drop=True)

    op = mae_scores[mae_scores["Type"] == "OP"].copy()
    op.drop(columns=["Type", "Panel"], inplace=True)
    plt.figure(figsize=(20, 10))
    sns.barplot(data=op, x="Marker", y="Score", hue="Segmentation", errorbar="sd")
    plt.title("MAE scores for OP biopsies\nComparison of different segmentation methods")
    plt.savefig(f"{save_folder}/op_mae_scores.png")

    data = op[op["Segmentation"] == "Unmicst + S3"].copy().reset_index(drop=True)
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
    plt.suptitle("Out Patient MAE Scores\nUnMICST + S3 Segmenter", x=0.45, fontsize=16)
    plt.title("Lower values indicate lower errors", x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_unmicst_s3_mae_scores_heatmap.png")

    data = op[op["Segmentation"] == "Mesmer"].copy().reset_index(drop=True)
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
    plt.suptitle("Out Patient MAE Scores\nMesmer", x=0.45, fontsize=16)
    plt.title("Lower values indicate lower errors", x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_mesmer_mae_scores_heatmap.png")

    # calculate difference between mesmer and unmicst separation
    s3 = op[op["Segmentation"] == "Unmicst + S3"].copy().reset_index(drop=True)
    mesmer = op[op["Segmentation"] == "Mesmer"].copy().reset_index(drop=True)

    # combine s3 and mesmer datasets
    df = pd.merge(s3, mesmer, on=["Biopsy", "Marker"], suffixes=("_s3", "_mesmer"))
    # calculate difference between s3 and mesmer
    df["Difference"] = df["Score_s3"].values - df["Score_mesmer"].values

    data = df.copy()
    data.drop(columns=["Segmentation_s3", "Score_s3", "Score_mesmer", "Segmentation_mesmer"],
              inplace=True)
    data.rename(columns={"Difference": "Score"}, inplace=True)

    #print(data)
    #df = (data.Score < 0.1).mul(100).reset_index(name='C', drop=True)
    #print(df)

    data = create_heatmap_df(data)
    data = data.pivot(index="Biopsy", columns="Marker", values="Score")
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=-0.2, vmax=0.3, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.xlabel('Feature')
    plt.suptitle("Out Patient Performance Difference\nMesmer vs. UnMicst + S3", x=0.45, fontsize=16)
    plt.title(
        "Negative values indicate better performance using UnMICST + S3\nHigher values indicate better performance using Mesmer",
        x=0.5, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_difference_mesmer_s3.png")


    tmp = op.copy()
    tmp.sort_values(by=["Biopsy", "Marker"], inplace=True)
    tmp["Biopsy"] = tmp["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.violinplot(data=tmp, x="Marker", y="Score", hue="Segmentation")
    plt.title("Out Patient Performance Difference\nSpread between biopsies\nMesmer vs. UnMicst + S3")
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_difference_mesmer_s3_violin_plot.png")
    plt.close('all')

    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.lineplot(data=tmp, x="Marker", y="Score", hue="Biopsy", style="Segmentation")
    plt.title("Out Patient Performance Difference\nMesmer vs. UnMicst + S3")
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_difference_mesmer_s3_lineplot.png")
    plt.close('all')

    # Prepare data for s3 segmentation
    data = s3.copy()
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    data.sort_values(by=["Biopsy", "Marker"], inplace=True)
    data = data.pivot(index="Marker", columns="Biopsy", values="Score")

    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.clustermap(data=data, vmin=0, vmax=0.5)
    plt.suptitle("In Patient UnMICST + S3 Segmenter \nBiopsy vs. Feature")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_cluster_map_unmicst_s3.png")
    plt.close('all')

    # Prepare data for mesmer segmentation
    data = mesmer.copy()
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    data.sort_values(by=["Biopsy", "Marker"], inplace=True)
    data = data.pivot(index="Marker", columns="Biopsy", values="Score")

    fig = plt.figure(figsize=(13, 5), dpi=200)
    sns.clustermap(data=data, vmin=0, vmax=0.5)
    plt.suptitle("In Patient Mesmer Segmentation \nBiopsy vs. Feature")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/op_cluster_map_mesmer.png")
    plt.close('all')
