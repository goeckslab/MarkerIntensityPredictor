import os, argparse
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

save_path = Path("plots/ae/plots")
lineage_markers = ["CK14", "CK17", "CK19", "aSMA", "Ecad", "CK7"]
functional_markers = ["Ki67", "pERK", "RAD51", "CycD1", "PR", "EGFR", "pRB", "p21", "AR", "cPARP", "HER2"]
immune_markers = ["CD45"]
markers_dict = {
    "lineage": lineage_markers,
    "functional": functional_markers,
    "immune": immune_markers
}


def load_ae_scores(mode: str, replace_value: str, add_noise: str, spatial: int):
    all_scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    noise: int = 1 if add_noise == "noise" else 0
    all_scores = all_scores[all_scores["Type"] == mode]
    all_scores = all_scores[all_scores["Replace Value"] == replace_value]
    all_scores = all_scores[all_scores["Noise"] == noise]
    all_scores = all_scores[all_scores["FE"] == spatial]
    all_scores = all_scores[all_scores["HP"] == 0]
    return all_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ip", "exp"], default="ip")
    parser.add_argument("--replace_all", action="store_true", default=False,
                        help="Loads the scores where all marker are replaced")
    parser.add_argument("-m", "--markers", nargs='+')
    parser.add_argument("-rv", "--replace_value", choices=["mean", "zero"], default="mean")
    parser.add_argument("-an", "--an", action="store_true", default=False)
    parser.add_argument("-sp", "--spatial", choices=[0, 23, 42, 92, 138, 184], default=0, type=int)
    args = parser.parse_args()

    replace_all = args.replace_all
    mode = args.mode
    markers = args.markers
    replace_value = args.replace_value
    add_noise = "noise" if args.an else "no_noise"
    spatial = args.spatial

    save_path = Path(save_path, "iterations")
    save_path = Path(save_path, mode)
    save_path = Path(save_path, str(spatial))

    if replace_all:
        save_path = Path(save_path, "replace_all")
    else:
        save_path = Path(save_path, "replace_one")

    if markers:
        save_path = Path(save_path, "_".join(markers))
    else:
        save_path = Path(save_path, "all_markers")

    save_path = Path(save_path, replace_value)
    save_path = Path(save_path, add_noise)

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    scores: pd.DataFrame = load_ae_scores(mode=mode, replace_value=replace_value, add_noise=add_noise, spatial=spatial)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 5), dpi=200)
    row = 0
    column = 0

    biopsies = scores["Biopsy"].unique()
    # sort biopsy names
    biopsies = sorted(biopsies)
    # create linechart for markers and iterations
    for biopsy in biopsies:

        biopsy_data = scores[scores["Biopsy"] == biopsy]
        sns.lineplot(data=biopsy_data, x="Iteration", y="MAE", hue="Marker", ax=axes[row, column])
        # sns.lineplot(data=biopsy_data, x="Iteration", y="MAE", hue="Marker", errorbar="sd", ax=axes[row, column])
        axes[row, column].set_title(biopsy.replace("_", " "))
        axes[row, column].set_xlabel("Iteration")
        axes[row, column].set_ylabel("MAE")
        axes[row, column].legend().set_visible(False)
        axes[row, column].set_ylim(0, 0.6)
        handles, labels = axes[row, column].get_legend_handles_labels()
        column += 1

        if column == 3:
            row += 1
            column = 0

        if column == 2 and row == 2:
            fig.delaxes(axes[row, column])

    # set fig title
    fig.suptitle(f"MAE per marker per iteration\n{mode.upper()}")
    fig.tight_layout()

    fig.legend(handles, labels, loc='lower right', ncol=3)
    if args.markers:
        plt.savefig(Path(save_path, f"ae_iterations_{mode}{'_replace_all' if replace_all else ''}.png"))
    else:
        plt.savefig(Path(save_path, f"ae_iterations_{mode}{'_replace_all' if replace_all else ''}_all_markers.png"))

    # create charts by lineage or functional

    print(markers_dict)

    for identification, markers in markers_dict.items():
        # select all markers
        scores_markers = scores[scores["Marker"].isin(markers)]
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 5), dpi=200)
        row = 0
        column = 0

        biopsies = scores_markers["Biopsy"].unique()
        # sort biopsy names
        biopsies = sorted(biopsies)
        # create linechart for markers and iterations
        for biopsy in biopsies:

            biopsy_data = scores_markers[scores_markers["Biopsy"] == biopsy]
            sns.lineplot(data=biopsy_data, x="Iteration", y="MAE", hue="Marker", ax=axes[row, column])
            # sns.lineplot(data=biopsy_data, x="Iteration", y="MAE", hue="Marker", errorbar="sd", ax=axes[row, column])
            axes[row, column].set_title(biopsy.replace("_", " "))
            axes[row, column].set_xlabel("Iteration")
            axes[row, column].set_ylabel("MAE")
            axes[row, column].legend().set_visible(False)
            axes[row, column].set_ylim(0, 0.6)
            handles, labels = axes[row, column].get_legend_handles_labels()
            column += 1

            if column == 3:
                row += 1
                column = 0

            if column == 2 and row == 2:
                fig.delaxes(axes[row, column])

        # set fig title
        fig.suptitle(f"MAE per marker per iteration\n{mode.replace('_', '').upper()}\n{identification.upper()}")
        fig.tight_layout()

        fig.legend(handles, labels, loc='lower right', ncol=3)
        if args.markers:
            plt.savefig(
                Path(save_path, f"{identification}.png"))
        else:
            plt.savefig(
                Path(save_path, f"{identification}_all_markers.png"))
