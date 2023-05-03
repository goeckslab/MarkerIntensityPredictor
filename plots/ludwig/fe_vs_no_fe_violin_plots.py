import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path
import argparse
from typing import List
import shutil
from statannotations.Annotator import Annotator

# Create violin plots showing the difference between feature engineering and no feature engineering
# Specific for a single spatial distance

biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]

ludwig_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig")
sp_23_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig_sp_23")
sp_46_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig_sp_46")
sp_92_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig_sp_92")
sp_138_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig_sp_138")
sp_184_ip_source_path = Path("data/scores/Mesmer/in_patient/Ludwig_sp_184")

ludwig_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig")
sp_23_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig_sp_23")
sp_46_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig_sp_46")
sp_92_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig_sp_92")
sp_138_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig_sp_138")
sp_184_op_source_path = Path("data/scores/Mesmer/out_patient/Ludwig_sp_184")

ip_folder = Path("ip_plots")
op_folder = Path("op_plots")
save_path = Path("plots/ludwig/fe_vs_no_fe")


def create_boxen_plot_per_segmentation(data: pd.DataFrame, score: str, title: str, save_folder: Path, file_name: str,
                                       ylim: List, color_palette, spatial_distance: str):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    # ax = sns.violinplot(data=data, x="Marker", y=score, hue="FE", split=True, cut=0, palette=color_palette)
    ax = sns.boxenplot(data=data, x="Marker", y=score, hue="FE", palette=color_palette)

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    # set y ticks of fig
    if args.markers:
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)
        ax.set_xticklabels(x_ticks, rotation=0, fontsize=20)
    plt.box(False)
    # remove legend from fig
    plt.legend().set_visible(False)


    hue = "FE"
    hue_order = [spatial_distance, "None"]
    pairs = [
        (("pRB", spatial_distance), ("pRB", "None")),
        (("CD45", spatial_distance), ("CD45", "None")),
        (("CK19", spatial_distance), ("CK19", "None")),
        (("Ki67", spatial_distance), ("Ki67", "None")),
        (("aSMA", spatial_distance), ("aSMA", "None")),
        (("Ecad", spatial_distance), ("Ecad", "None")),
        (("PR", spatial_distance), ("PR", "None")),
        (("CK14", spatial_distance), ("CK14", "None")),
        (("HER2", spatial_distance), ("HER2", "None")),
        (("AR", spatial_distance), ("AR", "None")),
        (("CK17", spatial_distance), ("CK17", "None")),
        (("p21", spatial_distance), ("p21", "None")),
        (("Vimentin", spatial_distance), ("Vimentin", "None")),
        (("pERK", spatial_distance), ("pERK", "None")),
        (("EGFR", spatial_distance), ("EGFR", "None")),
        (("ER", spatial_distance), ("ER", "None")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_line_plot(data: pd.DataFrame, metric: str):
    fig = plt.figure(dpi=200, figsize=(10, 6))
    ax = sns.lineplot(x="Marker", y=metric, hue="FE", data=data)
    plt.show()


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
    # argsparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', help="Markers to be plotted", default=None)
    parser.add_argument("-sp", "--spatial", type=int, help="Spatial distance", default=46, choices=[23, 46, 92, 138, 184])
    parser.add_argument("-t", "--type", type=str, default="ip", choices=["ip", "op"])
    parser.add_argument("-s", "--score", type=str, default="MAE", choices=["RMSE", "MAE"])
    args = parser.parse_args()

    # load mesmer mae scores from data mesmer folder and all subfolders
    spatial_distance = args.spatial
    patient_type: str = args.type
    metric: str = args.score
    markers = args.markers

    save_path = Path(save_path, patient_type)
    save_path = Path(save_path, str(spatial_distance))

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    if save_path.exists():
        shutil.rmtree(save_path)

    print(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    if patient_type == "ip":
        if args.spatial == 46:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_46_ip_source_path))
            my_pal = {"None": "grey", "sp_46": "purple"}
        elif args.spatial == 92:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_92_ip_source_path))
            my_pal = {"None": "grey", "sp_92": "green"}
        elif args.spatial == 138:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_138_ip_source_path))
            my_pal = {"None": "grey", "sp_138": "orange"}
        elif args.spatial == 184:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_184_ip_source_path))
            my_pal = {"None": "grey", "sp_184": "blue"}
        else:
            raise ValueError("Spatial distance not supported")
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_ip_source_path))
    else:
        if args.spatial == 46:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_46_op_source_path))
            my_pal = {"None": "grey", "sp_46": "purple"}
        elif args.spatial == 92:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_92_op_source_path))
            my_pal = {"None": "grey", "sp_92": "green"}
        elif args.spatial == 138:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_138_op_source_path))
            my_pal = {"None": "grey", "sp_138": "orange"}
        elif args.spatial == 184:
            fe_scores: pd.DataFrame = load_scores(source_path=str(sp_184_op_source_path))
            my_pal = {"None": "grey", "sp_184": "blue"}
        else:
            raise ValueError("Spatial distance not supported")
        no_fe_scores: pd.DataFrame = load_scores(source_path=str(ludwig_op_source_path))

    # combine both dataframes
    scores = pd.concat([fe_scores, no_fe_scores], axis=0)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    # Create bar plot which compares in patient performance of the different segementations for each biopsy
    # The bar plot should be saved in the plots folder

    file_name = f"{spatial_distance}_vs_no_fe"

    if args.markers:
        y_lim = [0, 0.3]
    else:
        y_lim = [0, 0.5]

    create_boxen_plot_per_segmentation(data=scores, score=metric.upper(),
                                       title=f"In & Out patient performance using spatial feature engineering",
                                       file_name=file_name, save_folder=save_path, ylim=y_lim, color_palette=my_pal,
                                       spatial_distance=f"sp_{spatial_distance}")

    # create_line_plot(data=scores, metric=metric.upper())
