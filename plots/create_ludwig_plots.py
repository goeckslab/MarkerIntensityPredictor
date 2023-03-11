import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]

ip_folder = Path("ip_plots")
op_folder = Path("op_plots")


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


def plot_performance_heatmap_per_segmentation(data, score: str, folder: Path, file_name: str, title: str):
    data = create_heatmap_df(data, score, decimal_precision=4)

    data = data.pivot(index="Biopsy", columns="Marker", values=score)
    data = data.loc[[f"{biopsy}" for biopsy in biopsies]]

    save_folder = Path(f"{folder}/ludwig")

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(data=data, vmin=0, vmax=0.5, annot=True)
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.title(title)
    plt.xlabel("Biopsy")
    plt.ylabel(score)
    plt.tight_layout()
    plt.savefig(Path(f"{save_folder}/{file_name}.png"))
    plt.close('all')


if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    scores = []
    for root, dirs, files in os.walk("data/scores"):
        for name in files:
            if Path(name).suffix == ".csv" and "_Ludwig_0" in name:
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 48, "Not all biopsies have been processed"
    # print(scores)

    scores = pd.concat(scores, axis=0)
    # Create bar plot which compares in patient performance of the different segementations for each biopsy
    # The bar plot should be saved in the plots folder

    ip_mae_scores = scores[scores["Type"] == "IP"].copy()

    mae_performance_data = ip_mae_scores.copy()
    mae_performance_data.drop(columns=["Type", "FE", "Mode", "Hyper", "Panel"], inplace=True)

    # Plot mesmer
    mae_performance_data_mesmer = mae_performance_data[mae_performance_data["Segmentation"] == "Mesmer"].copy()
    mae_performance_data_mesmer.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MAE", folder=ip_folder,
                                              file_name="ludwig_ip_mesmer_mae_heatmap",
                                              title=f"In Patient MAE Scores \n Mesmer")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MSE", folder=ip_folder,
                                              file_name="ludwig_ip_mesmer_mse_heatmap",
                                              title=f"In Patient MSE Scores \n Mesmer")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "RMSE", folder=ip_folder,
                                              file_name=
                                              "ludwig_ip_mesmer_rmse_heatmap",
                                              title=f"In Patient RMSE Scores \n Mesmer")

    # Plot unmicst snr
    mae_performance_data_s3_snr = mae_performance_data[
        (mae_performance_data["Segmentation"] == "Unmicst + S3") & (mae_performance_data["SNR"] == 1)].copy()

    mae_performance_data_s3_snr.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "MAE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_snr_mae_heatmap",
                                              title=f"In Patient MAE Scores \n UnMicst + S3 (SNR)")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "MSE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_snr_mse_heatmap",
                                              title=f"In Patient MSE Scores \n UnMicst + S3 (SNR)")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "RMSE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_snr_rmse_heatmap",
                                              title=f"In Patient RMSE Scores \n UnMicst + S3 (SNR)")

    mae_performance_data_s3_non_snr = mae_performance_data[
        (mae_performance_data["Segmentation"] == "Unmicst + S3") & (mae_performance_data["SNR"] == 0)].copy()
    mae_performance_data_s3_non_snr.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "MAE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_non_snr_mae_heatmap",
                                              title=f"In Patient MAE Scores \n UnMicst + S3 (Non SNR)")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "MSE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_non_snr_mse_heatmap",
                                              title=f"In Patient MSE Scores \n UnMicst + S3 (Non SNR)")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "RMSE",
                                              folder=ip_folder, file_name=
                                              "ludwig_ip_s3_non_snr_rmse_heatmap",
                                              title=f"In Patient RMSE Scores \n UnMicst + S3 (Non SNR)")

    # Out Patient

    op_mae_scores = scores[scores["Type"] == "OP"].copy()

    mae_performance_data = op_mae_scores.copy()
    mae_performance_data.drop(columns=["Type", "FE", "Mode", "Hyper", "Panel"], inplace=True)

    # Plot mesmer
    mae_performance_data_mesmer = mae_performance_data[mae_performance_data["Segmentation"] == "Mesmer"].copy()
    mae_performance_data_mesmer.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MAE", folder=op_folder,
                                              file_name="ludwig_op_mesmer_mae_heatmap",
                                              title=f"Out Patient MAE Scores \n Mesmer")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "MSE",
                                              title=f"Out Patient MSE Scores \n Mesmer", folder=op_folder,
                                              file_name="ludwig_op_mesmer_mse_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_mesmer, "RMSE",
                                              title=f"Out Patient RMSE Scores \n Mesmer", folder=op_folder,
                                              file_name=
                                              "ludwig_op_mesmer_rmse_heatmap")

    # Plot unmicst snr
    mae_performance_data_s3_snr = mae_performance_data[
        (mae_performance_data["Segmentation"] == "Unmicst + S3") & (mae_performance_data["SNR"] == 1)].copy()

    mae_performance_data_s3_snr.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "MAE",
                                              title="Out Patient MAE Scores \n UnMicst + S3 (SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_snr_mae_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "MSE",
                                              title="Out Patient MSE Scores \n UnMicst + S3 (SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_snr_mse_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_snr, "RMSE",
                                              title="Out Patient RMSE Scores \n UnMicst + S3 (SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_snr_rmse_heatmap")

    mae_performance_data_s3_non_snr = mae_performance_data[
        (mae_performance_data["Segmentation"] == "Unmicst + S3") & (mae_performance_data["SNR"] == 0)].copy()
    mae_performance_data_s3_non_snr.drop(columns=["SNR"], inplace=True)
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "MAE",
                                              title="Out Patient MAE Scores \n UnMicst + S3 (Non SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_non_snr_mae_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "MSE",
                                              title="Out Patient MSE Scores \n UnMicst + S3 (Non SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_non_snr_mse_heatmap")
    plot_performance_heatmap_per_segmentation(mae_performance_data_s3_non_snr, "RMSE",
                                              title="Out Patient RMSE Scores \n UnMicst + S3 (Non SNR)",
                                              folder=op_folder, file_name=
                                              "ludwig_op_s3_non_snr_rmse_heatmap")
