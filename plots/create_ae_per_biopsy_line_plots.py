import argparse, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ip_source_folder = "ae/ip"
op_source_folder = "ae/op"
ip_source_folder_spatial_46 = "ae_sp_46/ip"
op_source_folder_spatial_46 = "ae_sp_46/op"
ip_save_path = Path("ip_plots/ae/")
op_save_path = Path("op_plots/ae/")


def load_scores(source_folder: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file == "scores.csv":
                print("Loading file: ", file)
                scores.append(pd.read_csv(os.path.join(root, file), sep=",", header=0))

    assert len(scores) == 8, f"Not all biopsies have been selected. Only {len(scores)} biopsies have been selected."

    return pd.concat(scores, axis=0).sort_values(by=["Marker"]).reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    parser.add_argument("-sp", "--spatial", type=int, default=0, choices=[0, 46])
    parser.add_argument("-t", "--type", choices=["ip", "op"], default="ip")
    args = parser.parse_args()

    metric: str = args.scores
    patient_type: str = args.type
    spatial: int = args.spatial if args.spatial else 0

    if patient_type == "ip":
        if spatial == 46:
            save_path = ip_save_path / "sp_46"
            source_folder = ip_source_folder_spatial_46
        else:
            save_path = ip_save_path
            source_folder = ip_source_folder
    else:
        if spatial == 46:
            save_path = op_save_path / "sp_46"
            source_folder = op_source_folder_spatial_46
        else:
            save_path = op_save_path
            source_folder = op_source_folder

    print(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)



    scores = load_scores(source_folder=source_folder)
    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    scores = pd.melt(scores, id_vars=["Biopsy", "Marker"], var_name="Metric", value_name="Score")
    scores = scores[scores["Metric"] == metric.upper()].copy()
    # replace _ in biopsies names with spce
    scores["Biopsy"] = scores["Biopsy"].str.replace("_", " ")

    palette_colors = sns.color_palette('tab10')
    palette_dict = {continent: color for continent, color in zip(scores["Biopsy"].unique(), palette_colors)}

    # Create line plot separated by spatial resolution using seaborn
    fig = plt.figure(dpi=200, figsize=(10, 6))
    ax = sns.lineplot(x="Marker", y="Score", hue="Biopsy", data=scores, palette=palette_dict)
    handles, labels = ax.get_legend_handles_labels()

    points = zip(labels, handles)
    points = sorted(points)
    labels = [point[0] for point in points]
    handles = [point[1] for point in points]
    # change title of legend
    plt.legend(title="Biopsy", handles=handles, labels=labels)
    plt.xlabel("Markers")
    plt.ylabel(metric.upper())
    if args.markers:
        plt.ylim(0, 0.6)
    else:
        plt.ylim(0, 0.6)
    plt.title(
        f"{metric.upper()} scores\nDenoising performance for biopsies")
    plt.tight_layout()
    if args.markers:
        plt.savefig(
            f"{save_path}/denoising_{metric.lower()}_{str(spatial) + '_' if spatial != 0 else ''}ae_biopsies.png")
    else:
        plt.savefig(
            f"{save_path}/denoising_{metric.lower()}_{str(spatial) + '_' if spatial != 0 else ''}ae_biopsies_all_markers.png")

    plt.close('all')
