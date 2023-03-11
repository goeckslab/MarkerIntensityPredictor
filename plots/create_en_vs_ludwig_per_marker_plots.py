import os, shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
ip_folder = Path("ip_plots")
op_folder = Path("op_plots")


def create_bar_plot_for_marker_segmentations(data: pd.DataFrame, y: str, title: str, folder: Path, file_name: str):
    save_path = Path(f"{folder}/marker_segmentations")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    origin = data["Origin"].unique()
    origin.sort()
    palette_colors = sns.color_palette('tab10')
    palette_dict = {origin: color for origin, color in zip(origin, palette_colors)}

    # plot bar plot for ip and op en vs ludwig for all markers
    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.scatterplot(data=data, x="Biopsy", y=y, hue="Origin", style="Mode", palette=palette_dict)
    plt.ylim(0, 1)
    plt.xlabel("Biopsy")
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
    op_scores = scores[scores["Type"] == "OP"].copy()
    ip_scores = scores[scores["Type"] == "IP"].copy()

    op_scores["Origin"] = op_scores.apply(
        lambda x: "Mesmer" if "Mesmer" in x["Segmentation"] else "UnMICST + S3 SNR Corrected",
        axis=1)
    # rename origin to unmicst + s3 Non SNR if not snr corrected
    op_scores.loc[op_scores["SNR"] == 0, "Origin"] = "UnMICST + S3 Non SNR Corrected"

    ip_scores["Origin"] = ip_scores.apply(
        lambda x: "Mesmer" if "Mesmer" in x["Segmentation"] else "UnMICST + S3 SNR Corrected",
        axis=1)
    # rename origin to unmicst + s3 Non SNR if not snr corrected
    ip_scores.loc[ip_scores["SNR"] == 0, "Origin"] = "UnMICST + S3 Non SNR Corrected"

    # Create plots for each mode for each marker
    # for mode in ip_scores["Mode"].unique():
    #    sub_data = ip_scores[ip_scores["Mode"] == mode].copy()
    #
    #        for marker in sub_data["Marker"].unique():
    #           sub_data_marker = sub_data[sub_data["Marker"] == marker].copy()
    #          create_bar_plot_for_marker_segmentations(sub_data_marker, "MAE", f"IP {marker} scores for {mode}",
    #                      ip_folder, f"{marker}_{mode}")

    for marker in ip_scores["Marker"].unique():
        sub_data_marker = ip_scores[ip_scores["Marker"] == marker].copy()
        create_bar_plot_for_marker_segmentations(sub_data_marker, "MAE", f"IP {marker} scores",
                                                 ip_folder, f"{marker}")
