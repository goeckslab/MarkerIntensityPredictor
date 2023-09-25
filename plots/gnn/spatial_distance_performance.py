# Creates a linegraph which plots the radius and the performance for markers
from pathlib import Path
import argparse, shutil, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

save_path = Path("plots/gnn/spatial_distance_performance")


def load_gnn_scores(mode: str, replace_value: str, add_noise: str) -> pd.DataFrame:
    all_scores = pd.read_csv(Path("data", "scores", "gnn", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Mode"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    return all_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", type=str, default="ip", choices=["ip", "exp"])
    parser.add_argument("--metric", type=str, default="MAE", choices=["RMSE", "MAE"])
    parser.add_argument("-rv", "--replace_value", choices=["mean", "zero"], default="mean")
    args = parser.parse_args()

    mode: str = args.mode
    metric: str = args.metric
    markers = args.markers
    replace_value: str = args.replace_value

    save_path = Path(save_path, mode)
    save_path = Path(save_path, replace_value)

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if mode == "ip":
        scores = load_gnn_scores(mode="ip", replace_value=replace_value, add_noise="no_noise")

    else:
        scores = load_gnn_scores(mode="exp", replace_value=replace_value, add_noise="no_noise")

    # select best performing experiment üer
    scores = scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    scores = scores.groupby(["Marker", "Biopsy"]).first().reset_index()

    scores.replace(to_replace={"FE": {0: 0}}, inplace=True)
    scores.replace(to_replace={"FE": {23: 15}}, inplace=True)
    scores.replace(to_replace={"FE": {46: 30}}, inplace=True)
    scores.replace(to_replace={"FE": {92: 60}}, inplace=True)
    scores.replace(to_replace={"FE": {138: 90}}, inplace=True)
    scores.replace(to_replace={"FE": {184: 120}}, inplace=True)

    cat_size_order = CategoricalDtype(
        [0, 15, 30, 60, 90, 120],
        ordered=True
    )
    scores['FE'] = scores['FE'].astype(cat_size_order)
    scores.sort_values(by=["FE"], inplace=True)

    temp = scores.groupby(["Marker", "FE"]).mean().reset_index()

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.lineplot(data=temp, x="FE", y=metric, hue="Marker", markers=True, dashes=False)
    # Change x axis
    plt.xlabel("Distance (µm)")
    plt.title(f"Spatial Distance Performance\nMean of all Biopsies\n{replace_value} replacement")
    plt.ylim(0, 0.4)
    # move legend
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(Path(save_path, f"{metric}_spatial_performance.png"), bbox_inches='tight')
    plt.close('all')

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.boxenplot(data=scores, x="Marker", y=metric, hue="FE")
    # Change x axis
    plt.xlabel("Marker")
    plt.title("Spatial Distance Performance\nPerformance for each marker")
    plt.xticks([0, 15, 30, 60, 90, 120], [0, 15, 30, 60, 90, 120])
    plt.savefig(Path(save_path, f"{metric}_spatial_performance_boxen.png"), bbox_inches='tight')
    plt.close('all')
