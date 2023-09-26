# Creates a linegraph which plots the radius and the performance for markers
from pathlib import Path
import argparse, shutil, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

save_path = Path("plots/ae/plots/spatial_comparison")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", type=str, default="ip", choices=["ip", "exp"])
    parser.add_argument("--metric", type=str, default="MAE", choices=["RMSE", "MAE"])
    parser.add_argument("--replace_value", type=str, choices=["mean", "zero"], default="mean")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    parser.add_argument("-hp", "--hyper", action="store_true", required=False, default=False,
                        help="Should hyper parameter tuning be used?")
    args = parser.parse_args()

    mode: str = args.mode
    metric: str = args.metric
    markers = args.markers
    replace_value = args.replace_value
    hyper = args.hyper
    add_noise: str = "noise" if args.an else "no_noise"

    save_path = Path(save_path, mode)
    save_path = Path(save_path, replace_value)

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if add_noise:
        save_folder = Path(save_path, "noise")
    else:
        save_folder = Path(save_path, "no_noise")

    if hyper:
        save_folder = Path(save_folder, "hyper")
    else:
        save_folder = Path(save_folder, "no_hyper")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # load scores
    scores = pd.read_csv(str(Path("data", "scores", "ae", "scores.csv")))

    # Select subset
    scores = scores[scores["Type"] == mode]
    scores = scores[scores["Replace Value"] == replace_value]
    scores = scores[scores["Noise"] == 1 if 'noise' in add_noise else 0]

    scores = scores.sort_values(by=["Marker", "Biopsy", "MAE"])
    scores = scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).head(5)
    scores = scores.groupby(["Marker", "Biopsy", "FE", "Experiment"]).mean().reset_index()

    temp = scores.groupby(["Marker", "FE"]).mean(numeric_only=True).reset_index()

    # rename None in the Fe column to No FE
    temp.replace(to_replace={"FE": {0: 0}}, inplace=True)
    temp.replace(to_replace={"FE": {23: 15}}, inplace=True)
    temp.replace(to_replace={"FE": {46: 30}}, inplace=True)
    temp.replace(to_replace={"FE": {92: 60}}, inplace=True)
    temp.replace(to_replace={"FE": {138: 90}}, inplace=True)
    temp.replace(to_replace={"FE": {184: 120}}, inplace=True)

    cat_size_order = CategoricalDtype(
        [0, 15, 30, 60, 90, 120],
        ordered=True
    )
    temp['FE'] = temp['FE'].astype(cat_size_order)
    temp.sort_values(by=["FE"], inplace=True)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.lineplot(data=temp, x="FE", y=metric, hue="Marker", markers=True, dashes=False)
    # Change x axis
    plt.xlabel("Distance (µm)")

    plt.ylim(0, 0.25)

    # change x ticks
    plt.xticks([0, 15, 30, 60, 90, 120], [0, 15, 30, 60, 90, 120])
    # move legend
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(Path(save_path, f"{metric}_spatial_performance.png"), bbox_inches='tight')
