import pandas as pd
import os, argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ip_source_folder = "data/scores/Mesmer/in_patient"
op_source_folder = "data/scores/Mesmer/out_patient"
ip_save_path = Path("ip_plots/en_vs_ludwig_vs_hyper/")
op_save_path = Path("op_plots/en_vs_ludwig_vs_hyper/")


def load_scores(source_folder: str) -> pd.DataFrame:
    scores = []
    for root, dirs, _ in os.walk(source_folder):
        for directory in dirs:
            if directory == "Ludwig" or directory == "EN" or directory == "Ludwig_Hyper":
                print("Loading folder: ", directory)
                # load only files of this folder
                for sub_root, _, files in os.walk(os.path.join(root, directory)):
                    for name in files:
                        if Path(name).suffix == ".csv":
                            print("Loading file: ", name)
                            scores.append(pd.read_csv(os.path.join(sub_root, name), sep=",", header=0))

    assert len(scores) == 24, "Not all biopsies have been selected"

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    parser.add_argument("-t", "--type", choices=["ip", "op"], default="ip")
    args = parser.parse_args()
    metric: str = args.scores
    patient_type: str = args.type

    if patient_type == "ip":
        save_path = ip_save_path
        source_folder = ip_source_folder
    else:
        save_path = op_save_path
        source_folder = op_source_folder

    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load mesmer mae scores from data mesmer folder and all subfolders
    scores: pd.DataFrame = load_scores(source_folder=source_folder)
    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # convert hyper column to string
    scores["Hyper"] = scores["Hyper"].astype(str)

    # create new column model and use Mode and Hyper column as input
    scores["Model"] = scores["Mode"] + "_" + scores["Hyper"]
    # rename model Ludwig_1 to Ludwig_Hyper
    scores.loc[scores["Model"] == "Ludwig_1", "Model"] = "Ludwig Hyper"
    scores.loc[scores["Model"] == "Ludwig_0", "Model"] = "Ludwig"
    scores.loc[scores["Model"] == "EN_0", "Model"] = "EN"

    le = LabelEncoder()
    scores["Model Enc"] = le.fit_transform(scores["Model"])
    # create new column using the model columns and translate the model column to numbers

    scores = scores[["MAE", "RMSE", "Biopsy", "Marker", "Model", "Model Enc"]].copy()

    # melt scores keep markers and spatial resolution
    scores = pd.melt(scores, id_vars=["Biopsy", "Model", "Marker", "Model Enc"], var_name="Metric", value_name="Score")

    scores = scores[scores["Metric"] == metric.upper()].copy()

    palette_colors = sns.color_palette('tab10')
    palette_dict = {continent: color for continent, color in zip(scores["Model Enc"].unique(), palette_colors)}

    for biopsy in scores["Biopsy"].unique():
        data = scores[scores["Biopsy"] == biopsy]

        # Create line plot separated by spatial resolution using seaborn
        fig = plt.figure(dpi=200, figsize=(10, 6))
        ax = sns.lineplot(x="Marker", y="Score", hue="Model Enc", data=data, palette=palette_dict)
        plt.title(f"{metric} scores for biopsy {biopsy.replace('_', ' ')}")
        handles, labels = ax.get_legend_handles_labels()

        # convert labels to int
        temp_labels = [int(label) for label in labels]

        # convert labels to int
        labels = le.inverse_transform(temp_labels)

        points = zip(labels, handles)
        points = sorted(points)
        labels = [point[0] for point in points]
        handles = [point[1] for point in points]
        # change title of legend
        plt.legend(title="Model", handles=handles, labels=labels)
        plt.xlabel("Markers")
        plt.ylabel(metric)
        plt.ylim(0, 0.4)
        plt.title(
            f"{metric.upper()} scores for biopsy {biopsy.replace('_', ' ')}\nPerformance difference between base, non-linear and hyper models")
        plt.tight_layout()
        if args.markers:
            plt.savefig(f"{save_path}/{metric.lower()}_scores_{biopsy}_en_ludwig_hyper.png")
        else:
            plt.savefig(f"{save_path}/{metric.lower()}_scores_{biopsy}_en_ludwig_hyper_all_markers.png")

        plt.close('all')

    # create new df with only the means of each model for each marker
    # data = scores.groupby(["Marker", "Model Enc"]).mean(numeric_only=True).reset_index()

    fig = plt.figure(dpi=200, figsize=(10, 6))
    ax = sns.lineplot(x="Marker", y="Score", hue="Model Enc", data=scores, palette=palette_dict)
    plt.title(f"Mean {metric.upper()} scores")
    handles, labels = ax.get_legend_handles_labels()

    # convert labels to int
    temp_labels = [int(label) for label in labels]

    # convert labels to int
    labels = le.inverse_transform(temp_labels)

    points = zip(labels, handles)
    points = sorted(points)
    labels = [point[0] for point in points]
    handles = [point[1] for point in points]
    # change title of legend
    plt.legend(title="Model", handles=handles, labels=labels)
    plt.xlabel("Markers")
    plt.ylabel(metric.upper())
    plt.ylim(0, 0.4)
    plt.title(
        f"{metric.upper()} scores \nPerformance difference between base, non-linear and hyper models")
    plt.tight_layout()
    if args.markers:
        plt.savefig(f"{save_path}/{metric.lower()}_scores_en_ludwig_hyper_line.png")
    else:
        plt.savefig(f"{save_path}/{metric.lower()}_scores_en_ludwig_hyper_line_all_markers.png")

    plt.close('all')
