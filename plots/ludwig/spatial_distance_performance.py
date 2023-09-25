# Creates a linegraph which plots the radius and the performance for markers
from pathlib import Path
import argparse, shutil, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

ludwig_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig")
sp_23_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_23")
sp_46_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_46")
sp_92_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_92")
sp_138_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_138")
sp_184_ip_source_path = Path("data/scores/Mesmer/ip/Ludwig_sp_184")

ludwig_op_source_path = Path("data/scores/Mesmer/exp/Ludwig")
sp_23_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_23")
sp_46_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_46")
sp_92_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_92")
sp_138_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_138")
sp_184_op_source_path = Path("data/scores/Mesmer/exp/Ludwig_sp_184")

save_path = Path("plots/ludwig/fe_vs_no_fe")


def load_scores(source_path: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(source_path):
        for name in files:
            if Path(name).suffix == ".csv":
                print("Loading file: ", os.path.join(root, name))
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, f"Should have loaded 8 biopsies but loaded: {len(scores)}, for path {source_path}"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("--mode", type=str, default="ip", choices=["ip", "exp"])
    parser.add_argument("--metric", type=str, default="MAE", choices=["RMSE", "MAE"])
    args = parser.parse_args()

    mode: str = args.mode
    metric: str = args.metric
    markers = args.markers

    save_path = Path(save_path, mode)
    save_path = Path(save_path, "spatial_comparison")

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if mode == "ip":
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_ip_source_path))
        sp_23_scores = load_scores(source_path=str(sp_23_ip_source_path))
        sp_46_scores = load_scores(source_path=str(sp_46_ip_source_path))
        sp_92_scores = load_scores(source_path=str(sp_92_ip_source_path))
        sp_138_scores = load_scores(source_path=str(sp_138_ip_source_path))
        sp_184_scores = load_scores(source_path=str(sp_184_ip_source_path))

        # combine all scores
        scores = pd.concat([no_fe_scores, sp_23_scores, sp_46_scores, sp_92_scores, sp_138_scores, sp_184_scores],
                           axis=0).reset_index(drop=True)

    else:
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_op_source_path))
        sp_23_scores = load_scores(source_path=str(sp_23_op_source_path))
        sp_46_scores = load_scores(source_path=str(sp_46_op_source_path))
        sp_92_scores = load_scores(source_path=str(sp_92_op_source_path))
        sp_138_scores = load_scores(source_path=str(sp_138_op_source_path))
        sp_184_scores = load_scores(source_path=str(sp_184_op_source_path))

        # combine all scores
        scores = pd.concat([no_fe_scores, sp_23_scores, sp_46_scores, sp_92_scores, sp_138_scores, sp_184_scores],
                           axis=0).reset_index(drop=True)

    print(scores)

    temp = scores.groupby(["Marker", "FE"]).mean().reset_index()

    cat_size_order = CategoricalDtype(
        [0, 23, 46, 92, 138, 184],
        ordered=True
    )
    temp['FE'] = temp['FE'].astype(cat_size_order)
    temp.sort_values(by=["FE"], inplace=True)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    sns.lineplot(data=temp, x="FE", y=metric, hue="Marker", markers=True, dashes=False)
    # Change x axis
    plt.xlabel("Distance (µm)")

    plt.ylim(0, 0.25)
    # move legend
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(Path(save_path, f"{metric}_spatial_performance.png"), bbox_inches='tight')
