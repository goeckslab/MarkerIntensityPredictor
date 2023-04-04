import pandas as pd
from pathlib import Path
import os, argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

ip_source_folder = "data/scores/Mesmer/in_patient"
op_source_folder = "data/scores/Mesmer/out_patient"
save_path = Path("op_vs_ip_plots/ludwig_fe/fe_vs_no_fe_line_charts/")


def load_scores(source_folder: str) -> pd.DataFrame:
    scores = []
    for root, dirs, _ in os.walk(source_folder):
        for directory in dirs:
            if "Ludwig_sp_" in directory or directory == "Ludwig":
                print("Loading folder: ", directory)
                # load only files of this folder
                for sub_root, _, files in os.walk(os.path.join(root, directory)):
                    for name in files:
                        if Path(name).suffix == ".csv":
                            print("Loading file: ", name)
                            scores.append(pd.read_csv(os.path.join(sub_root, name), sep=",", header=0))

    assert len(scores) == 40, "Selected biopsies are not 40, but " + str(len(scores))

    return pd.concat(scores, axis=0).sort_values(by=["Marker"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    args = parser.parse_args()

    metric: str = args.scores
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load mesmer mae scores from data mesmer folder and all subfolders
    ip_scores: pd.DataFrame = load_scores(source_folder=ip_source_folder)

    op_scores: pd.DataFrame = load_scores(source_folder=op_source_folder)
    # Merge ip and op scores
    scores = pd.concat([ip_scores, op_scores], axis=0)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # Replace FE None with 0
    scores.loc[scores["FE"] == "None", "FE"] = "sp_0"

    # create a new column spatial resolution and fill it with the spatial resolution of the biopsy
    scores["Spatial"] = scores["FE"].apply(lambda x: x.split("_")[1])

    # convert spatial to string
    scores["Spatial"] = scores["Spatial"].astype(str)
    # convert hyper column to string
    scores["Hyper"] = scores["Hyper"].astype(str)

    # Create new column Model and use Mode and Hyper column as input
    scores["Model"] = scores["Mode"] + "_" + scores["Hyper"] + "_" + scores["Spatial"]

    # rename models
    scores.loc[scores["Model"] == "Ludwig_1_0", "Model"] = "Ludwig Hyper"
    scores.loc[scores["Model"] == "Ludwig_0_0", "Model"] = "Ludwig"
    scores.loc[scores["Model"] == "EN_0_0", "Model"] = "EN"
    scores.loc[scores["Model"] == "Ludwig_0_46", "Model"] = "46"
    scores.loc[scores["Model"] == "Ludwig_0_92", "Model"] = "92"
    scores.loc[scores["Model"] == "Ludwig_0_138", "Model"] = "138"
    scores.loc[scores["Model"] == "Ludwig_0_184", "Model"] = "184"

    # select column MAE, Biopsy and FE
    scores = scores[["MAE", "RMSE", "Biopsy", "Model", "Marker", "Type"]].copy()
    # melt scores keep markers and spatial resolution
    scores = pd.melt(scores, id_vars=["Biopsy", "Model", "Marker", "Type"], var_name="Metric", value_name="Score")

    # only select model Ludwig , 46, 92
    scores = scores[scores["Model"].isin(["Ludwig", "46", "92"])].copy()

    # scores = pd.melt(scores, id_vars=["Biopsy", "Spatial"], var_name="metric", value_name="score")
    # Select MAe scores
    scores = scores[scores["Metric"] == metric.upper()].copy()

    # convert model to string
    scores["Model"] = scores["Model"].astype(str)

    le = LabelEncoder()
    scores["Model Enc"] = le.fit_transform(scores["Model"])

    print(scores)
    palette_colors = sns.color_palette('tab10')
    palette_dict = {continent: color for continent, color in zip(scores["Model Enc"].unique(), palette_colors)}

    for biopsy in scores["Biopsy"].unique():
        data = scores[scores["Biopsy"] == biopsy]
        # Create line plot separated by spatial resolution using seaborn
        fig = plt.figure(dpi=200, figsize=(10, 6))
        ax = sns.lineplot(x="Marker", y="Score", hue="Model Enc", style="Type", data=data, palette=palette_dict)
        plt.title(f"MAE scores for biopsy {biopsy.replace('_', ' ')}")
        handles, labels = ax.get_legend_handles_labels()

        # select only labels to can be converted to int
        # convert labels to int
        temp_labels = [int(label) for label in labels if label.isnumeric()]
        # select labels that are not int
        str_labels = [label for label in labels if not label.isnumeric()]

        # convert labels to int
        dec_labels = le.inverse_transform(temp_labels)

        # insert dec labels at the second position of str labels
        str_labels[1:1] = dec_labels
        # points = zip(str_labels, handles)
        # points = sorted(points, key=lambda x: (x[0].isnumeric(), int(x[0]) if x[0].isnumeric() else x[0]))
        # labels = [point[0] for point in points]
        # handles = [point[1] for point in points]

        labels = str_labels
        # change title of legend
        plt.legend(title="Microns", handles=handles, labels=labels)
        plt.xlabel("Markers")
        plt.ylabel(metric.upper())
        plt.title(
            f"{metric.upper()} scores for biopsy {biopsy.replace('_', ' ')}\nPerformance difference between IP & OP and spatial features")
        plt.tight_layout()
        if args.markers:
            plt.savefig(f"{save_path}/{metric.lower()}_scores_{biopsy}.png")
        else:
            plt.savefig(f"{save_path}/{metric.lower()}_scores_{biopsy}_all_markers.png")

        plt.close('all')

    # create new df with only the means of each model for each marker
    # scores = scores.groupby(["Biopsy", "Model", "Marker", "Model Enc"]).mean().reset_index()
    #scores = scores.groupby(["Marker", "Model Enc", "Type"]).mean(numeric_only=True).reset_index()
    # Create line plot separated by spatial resolution using seaborn
    fig = plt.figure(dpi=200, figsize=(10, 6))
    # sns.violinplot(x="Marker", y="Score", hue="Model Enc", data=data, palette=palette_dict)
    ax = sns.lineplot(x="Marker", y="Score", hue="Model Enc", style="Type", data=scores, palette=palette_dict)

    handles, labels = ax.get_legend_handles_labels()

    # select only labels to can be converted to int
    # convert labels to int
    temp_labels = [int(label) for label in labels if label.isnumeric()]
    # select labels that are not int
    str_labels = [label for label in labels if not label.isnumeric()]

    # convert labels to int
    dec_labels = le.inverse_transform(temp_labels)

    # insert dec labels at the second position of str labels
    str_labels[1:1] = dec_labels
    labels = str_labels

    labels[1], labels[2], labels[3] = labels[3], labels[1], labels[2]
    handles[1], handles[2], handles[3] = handles[3], handles[1], handles[2]

    # points = zip(labels, handles)
    # points = sorted(points, key=lambda x: (x[0].isnumeric(), int(x[0]) if x[0].isnumeric() else x[0]))

    # labels = [point[0] for point in points]
    # handles = [point[1] for point in points]
    # change title of legend
    plt.legend(title="Model", handles=handles, labels=labels)
    plt.xlabel("Markers")
    plt.ylabel(metric.upper())
    if args.markers:
        plt.ylim(0.025, 0.2)
    else:
        plt.ylim(0.025, 0.29)
    plt.title(
        f"{metric.upper()} scores\nPerformance difference between IP & OP and spatial features")
    plt.tight_layout()
    if args.markers:
        plt.savefig(f"{save_path}/{metric.lower()}_scores_mean_fe_vs_ludwig.png")
    else:
        plt.savefig(f"{save_path}/{metric.lower()}_scores_mean_fe_vs_ludwig_all_markers.png")

    plt.close('all')
