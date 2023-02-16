import argparse, os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

save_folder = Path("op_vs_ip_plots")

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

    for root, dirs, files in os.walk("unmicst_s3_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_non_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(mae_scores) == 48, "There should be 48 mae scores files"
    mae_scores = pd.concat(mae_scores, axis=0).reset_index(drop=True)

    op_full = mae_scores[mae_scores["Type"] == "OP"].copy()
    ip_full = mae_scores[mae_scores["Type"] == "IP"].copy()

    op_mesmer = op_full[op_full["Segmentation"] == "Mesmer"].copy()
    op_unmicst_s3 = op_full[op_full["Segmentation"] == "Unmicst + S3"].copy()

    ip_mesmer = ip_full[ip_full["Segmentation"] == "Mesmer"].copy()
    ip_unmicst_s3_snr = ip_full[(ip_full["Segmentation"] == "Unmicst + S3") & (ip_full["SNR"] == 1)].copy()
    ip_unmicst_s3_non_snr = ip_full[(ip_full["Segmentation"] == "Unmicst + S3") & (ip_full["SNR"] == 0)].copy()

    mae_scores["Type"] = [type.replace("OP", "Out Patient") for type in list(mae_scores["Type"].values)]
    mae_scores["Type"] = [type.replace("IP", "In Patient") for type in list(mae_scores["Type"].values)]

    # Select only mesmer segmentation
    ip_vs_op_mesmer = mae_scores[mae_scores["Segmentation"] == "Mesmer"].copy()

    # Select only unmicst + s3 segmentation snr corrected
    ip_vs_op_unmicst_s3_snr = mae_scores[
        (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 1)].copy()

    # Select only unmicst + s3 segmentation non snr corrected
    ip_vs_op_unmicst_s3_non_snr = mae_scores[
        (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 0)].copy()

    fig = plt.figure(figsize=(15, 5), dpi=200)
    sns.violinplot(data=ip_vs_op_mesmer, x="Marker", y="Score", hue="Type", split=False, inner="point", palette="Set2")
    plt.title("Mesmer Segmentation\nOut Patient vs In Patient\nDistribution of biopsy performance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/mesmer_ip_vs_op_performance.png")
    plt.close('all')

    fig = plt.figure(figsize=(15, 5), dpi=200)
    sns.violinplot(data=ip_vs_op_unmicst_s3_snr, x="Marker", y="Score", hue="Type", split=False, inner="point",
                   palette="Set2")
    plt.title("UnMICST + S3 Segmentation SNR corrected\nOut Patient vs In Patient\nDistribution of biopsy performance")
    #plt.legend(bbox_to_anchor=(1.10, 1), loc='upper right')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/unmisct_s3_snr_ip_vs_op_performance.png")
    plt.close('all')

    fig = plt.figure(figsize=(15, 5), dpi=200)
    sns.violinplot(data=ip_vs_op_unmicst_s3_non_snr, x="Marker", y="Score", hue="Type", split=False, inner="point",
                   palette="Set2")
    plt.title(
        "UnMICST + S3 Segmentation Non SNR corrected\nOut Patient vs In Patient\nDistribution of biopsy performance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/unmisct_s3_non_snr_ip_vs_op_performance.png")
    plt.close('all')
