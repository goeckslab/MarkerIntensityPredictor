import shutil

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path
import numpy as np

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
ip_folder = Path("ip_plots")
op_folder = Path("op_plots")


def create_bar_plot_for_modes(data: pd.DataFrame, y: str, title: str, folder: Path, file_name: str):
    save_path = Path(f"{folder}/modes")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    modes = data["Mode"].unique()
    modes.sort()
    palette_colors = sns.color_palette('tab10')
    palette_dict = {mode: color for mode, color in zip(modes, palette_colors)}

    # plot bar plot for ip and op en vs ludwig for all markers
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.barplot(data=data, x="Marker", y=y, hue="Mode", palette=palette_dict)
    plt.ylim(0, 1)
    plt.xlabel("Marker")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(f"{save_path}/{file_name}.png"))
    plt.close('all')


def create_bar_plot_for_segmentations(data: pd.DataFrame, y: str, title: str, folder: Path, file_name: str):
    save_path = Path(f"{folder}/segmentations")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    origin = data["Origin"].unique()
    origin.sort()
    palette_colors = sns.color_palette('tab10')
    palette_dict = {origin: color for origin, color in zip(origin, palette_colors)}

    # plot bar plot for ip and op en vs ludwig for all markers
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.barplot(data=data, x="Marker", y=y, hue="Origin", palette=palette_dict)
    plt.ylim(0, 1)
    plt.xlabel("Marker")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(f"{save_path}/{file_name}.png"))
    plt.close('all')


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    scores = []
    for root, dirs, files in os.walk("data/scores"):
        for name in files:
            if Path(name).suffix == ".csv" and "_EN_" in name or "_Ludwig_0":
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 96, "Not all biopsies have been processed"

    scores = pd.concat(scores, axis=0)

    ip_en_mesmer = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Mesmer") & (scores["Mode"] == "EN")].copy()
    ip_en_s3_snr = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 1) & (
                scores["Mode"] == "EN")].copy()
    ip_en_s3_non_snr = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 0) & (
                scores["Mode"] == "EN")].copy()

    ip_ludwig_mesmer = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Mesmer") & (scores["Mode"] == "Ludwig_0")].copy()
    ip_ludwig_s3_snr = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 1) & (
                scores["Mode"] == "Ludwig_0")].copy()
    ip_ludwig_s3_non_snr = scores[
        (scores["Type"] == "IP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 0) & (
                scores["Mode"] == "Ludwig_0")].copy()

    ip_scores = scores[scores["Type"] == "IP"].copy()

    op_en_mesmer = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Mesmer") & (scores["Mode"] == "EN")].copy()
    op_en_s3_snr = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 1) & (
                scores["Mode"] == "EN")].copy()
    op_en_s3_non_snr = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 0) & (
                scores["Mode"] == "EN")].copy()

    op_ludwig_mesmer = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Mesmer") & (scores["Mode"] == "Ludwig_0")].copy()
    op_ludwig_s3_snr = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 1) & (
                scores["Mode"] == "Ludwig_0")].copy()
    op_ludwig_s3_non_snr = scores[
        (scores["Type"] == "OP") & (scores["Segmentation"] == "Unmicst + S3") & (scores["SNR"] == 0) & (
                scores["Mode"] == "Ludwig_0")].copy()

    op_scores = scores[scores["Type"] == "OP"].copy()

    compare_ip_performance = ip_scores.copy()
    # Rename unmicst +s3 to unmicst +s3 non snr where snr == 0
    compare_ip_performance["Origin"] = compare_ip_performance.apply(
        lambda x: "Mesmer" if "Mesmer" in x["Segmentation"] else "UnMICST + S3 SNR Corrected",
        axis=1)
    # rename origin to unmicst + s3 Non SNR if not snr corrected
    compare_ip_performance.loc[compare_ip_performance["SNR"] == 0, "Origin"] = "UnMICST + S3 Non SNR Corrected"

    # In person mode
    for segmentation in compare_ip_performance["Origin"].unique():
        sub_data = compare_ip_performance[compare_ip_performance["Origin"] == segmentation].copy()
        # sort sub data by mode
        sub_data.sort_values(by=["Mode"], inplace=True)
        create_bar_plot_for_modes(sub_data, "MAE",
                                  f"In Patient Evaluation\n{segmentation} average MAE over all biopsies\nPerformance difference by mode",
                                  ip_folder,
                                  f"ip_{segmentation}_average_mae")
        create_bar_plot_for_modes(sub_data, "MSE",
                                  f"In Patient Evaluation\n{segmentation} average MSE over all biopsies\nPerformance difference by mode",
                                  ip_folder,
                                  f"ip_{segmentation}_average_mse")
        create_bar_plot_for_modes(sub_data, "RMSE",
                                  f"In Patient Evaluation\n{segmentation} average RMSE over all biopsies\nPerformance difference by mode",
                                  ip_folder,
                                  f"ip_{segmentation}_average_rmse")

        for biopsy in sub_data["Biopsy"].unique():
            # create bar plots for each biopsy for each mode
            biopsy_data = sub_data[sub_data["Biopsy"] == biopsy].copy()
            biopsy_data.sort_values(by=["Marker", "Mode"], inplace=True)
            create_bar_plot_for_modes(biopsy_data, "MAE",
                                      f"In Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} MAE\nPerformance difference by mode",
                                      ip_folder,
                                      f"ip_{segmentation}_{biopsy}_mae")
            create_bar_plot_for_modes(biopsy_data, "MSE",
                                      f"In Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} MSE\nPerformance difference by mode",
                                      ip_folder,
                                      f"ip_{segmentation}_{biopsy}_mse")
            create_bar_plot_for_modes(biopsy_data, "RMSE",
                                      f"In Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} RMSE\nPerformance difference by mode",
                                      ip_folder,
                                      f"ip_{segmentation}_{biopsy}_rmse")

    # Create for each segmentation the average performance over all biopsies
    for mode in compare_ip_performance["Mode"].unique():
        sub_data = compare_ip_performance[compare_ip_performance["Mode"] == mode].copy()
        # sort sub data by mode
        sub_data.sort_values(by=["Origin"], inplace=True)
        create_bar_plot_for_segmentations(sub_data, "MAE",
                                          f"In Patient Evaluation\n{mode} average MAE over all biopsies\nPerformance difference by segmentation",
                                          ip_folder,
                                          f"ip_{mode}_average_mae")
        create_bar_plot_for_segmentations(sub_data, "MSE",
                                          f"In Patient Evaluation\n{mode} average MSE over all biopsies", ip_folder,
                                          f"ip_{mode}_average_mse")
        create_bar_plot_for_segmentations(sub_data, "RMSE",
                                          f"In Patient Evaluation\n{mode} average RMSE over all biopsies", ip_folder,
                                          f"ip_{mode}_average_rmse")

        for biopsy in sub_data["Biopsy"].unique():
            # create bar plots for each biopsy for each mode
            biopsy_data = sub_data[sub_data["Biopsy"] == biopsy].copy()
            biopsy_data.sort_values(by=["Marker", "Origin"], inplace=True)
            create_bar_plot_for_segmentations(biopsy_data, "MAE",
                                              f"In Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} MAE",
                                              ip_folder,
                                              f"ip_{mode}_{biopsy}_mae")
            create_bar_plot_for_segmentations(biopsy_data, "MSE",
                                              f"In Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} MSE",
                                              ip_folder,
                                              f"ip_{mode}_{biopsy}_mse")
            create_bar_plot_for_segmentations(biopsy_data, "RMSE",
                                              f"In Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} RMSE",
                                              ip_folder,
                                              f"ip_{mode}_{biopsy}_rmse")

    compare_op_performance = op_scores.copy()
    # Rename unmicst +s3 to unmicst +s3 non snr where snr == 0
    compare_op_performance["Origin"] = compare_op_performance.apply(
        lambda x: "Mesmer" if "Mesmer" in x["Segmentation"] else "UnMICST + S3 SNR Corrected",
        axis=1)
    # rename origin to unmicst + s3 Non SNR if not snr corrected
    compare_op_performance.loc[compare_op_performance["SNR"] == 0, "Origin"] = "UnMICST + S3 Non SNR Corrected"

    # Create plots for each mode
    for segmentation in compare_op_performance["Origin"].unique():
        sub_data = compare_op_performance[compare_op_performance["Origin"] == segmentation].copy()
        # sort sub data by mode
        sub_data.sort_values(by=["Mode"], inplace=True)
        create_bar_plot_for_modes(sub_data, "MAE",
                                  f"Out Patient Evaluation\n{segmentation} average MAE over all biopsies\nPerformance difference by mode",
                                  op_folder,
                                  f"op_{segmentation}_average_mae")
        create_bar_plot_for_modes(sub_data, "MSE",
                                  f"Out Patient Evaluation\n{segmentation} average MSE over all biopsies\nPerformance difference by mode",
                                  op_folder,
                                  f"op_{segmentation}_average_mse")
        create_bar_plot_for_modes(sub_data, "RMSE",
                                  f"Out Patient Evaluation\n{segmentation} average RMSE over all biopsies\nPerformance difference by mode",
                                  op_folder,
                                  f"op_{segmentation}_average_rmse")

        for biopsy in sub_data["Biopsy"].unique():
            # create bar plots for each biopsy for each mode
            biopsy_data = sub_data[sub_data["Biopsy"] == biopsy].copy()
            biopsy_data.sort_values(by=["Marker", "Mode"], inplace=True)
            create_bar_plot_for_modes(biopsy_data, "MAE",
                                      f"Out Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} MAE\nPerformance difference by mode",
                                      op_folder,
                                      f"op_{segmentation}_{biopsy}_mae")
            create_bar_plot_for_modes(biopsy_data, "MSE",
                                      f"Out Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} MSE\nPerformance difference by mode",
                                      op_folder,
                                      f"op_{segmentation}_{biopsy}_mse")
            create_bar_plot_for_modes(biopsy_data, "RMSE",
                                      f"Out Patient Evaluation\n{segmentation} {biopsy.replace('_', ' ')} RMSE\nPerformance difference by mode",
                                      op_folder,
                                      f"op_{segmentation}_{biopsy}_rmse")

    # Create for each segmentation the average performance over all biopsies
    for mode in compare_op_performance["Mode"].unique():
        sub_data = compare_op_performance[compare_op_performance["Mode"] == mode].copy()
        # sort sub data by mode
        sub_data.sort_values(by=["Origin"], inplace=True)
        create_bar_plot_for_segmentations(sub_data, "MAE",
                                          f"Out Patient Evaluation\n{mode} average MAE over all biopsies\nPerformance difference by segmentation",
                                          op_folder,
                                          f"op_{mode}_average_mae")
        create_bar_plot_for_segmentations(sub_data, "MSE",
                                          f"Out Patient Evaluation\n{mode} average MSE over all biopsies\nPerformance difference by segmentation",
                                          op_folder,
                                          f"op_{mode}_average_mse")
        create_bar_plot_for_segmentations(sub_data, "RMSE",
                                          f"Out Patient Evaluation\n{mode} average RMSE over all biopsies\nPerformance difference by segmentation",
                                          op_folder,
                                          f"op_{mode}_average_rmse")

        for biopsy in sub_data["Biopsy"].unique():
            # create bar plots for each biopsy for each mode
            biopsy_data = sub_data[sub_data["Biopsy"] == biopsy].copy()
            biopsy_data.sort_values(by=["Marker", "Origin"], inplace=True)
            create_bar_plot_for_segmentations(biopsy_data, "MAE",
                                              f"Out Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} MAE\nPerformance difference by segmentation",
                                              op_folder,
                                              f"op_{mode}_{biopsy}_mae")
            create_bar_plot_for_segmentations(biopsy_data, "MSE",
                                              f"Out Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} MSE\nPerformance difference by segmentation",
                                              op_folder,
                                              f"op_{mode}_{biopsy}_mse")
            create_bar_plot_for_segmentations(biopsy_data, "RMSE",
                                              f"Out Patient Evaluation\n{mode} {biopsy.replace('_', ' ')} RMSE\nPerformance difference by segmentation",
                                              op_folder,
                                              f"op_{mode}_{biopsy}_rmse")
