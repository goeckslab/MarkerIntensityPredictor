import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

rounds = {"9_2": {
    "2": ["pERK"],
    "3": ["Vimentin", "aSMA"],
    "4": ["Ecad", "ER", "PR"],
    "5": ["EGFR", "pRB", "CD45"],
    "6": ["Ki67", "CK19", "p21"],
    "7": ["CK14", "AR"],
    "8": ["CK17", "HER2"],
},
    "9_3": {
        "2": ["pERK"],
        "3": ["Vimentin", "aSMA"],
        "4": ["Ecad", "ER", "PR"],
        "5": ["EGFR", "pRB", "CD45"],
        "6": ["Ki67", "CK19", "p21"],
        "7": ["CK14", "AR"],
        "8": ["CK17", "HER2"],
    },
    "9_14": {
        "2": ["pERK"],
        "3": ["Ki67", "CD45"],
        "4": ["Ecad", "aSMA", "Vimentin"],
        "5": ["pRB", "EGFR", "p21"],
        "7": ["ER", "HER2"],
        "8": ["CK14", "CK19", "CK17"],
        "9": ["AR"],
        "10": ["PR"]
    },
    "9_15": {
        "2": ["pERK"],
        "3": ["Ki67", "CD45"],
        "4": ["Ecad", "aSMA", "Vimentin"],
        "5": ["pRB", "EGFR", "p21"],
        "7": ["ER", "HER2"],
        "8": ["CK14", "CK19", "CK17"],
        "9": ["AR"],
        "10": ["PR"]
    }
}

results_folder = Path("results/expression_by_round")

if __name__ == '__main__':

    if not results_folder.exists():
        results_folder.mkdir(parents=True)
    # Check whether marker expression is high

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", help='Input file')
    args = parser.parse_args()

    biopsy = args.biopsy
    biopsy_data = pd.read_csv(biopsy)
    segmentation = biopsy.split("/")[1]

    if segmentation == "tumor_mesmer":
        segmentation = "Mesmer"
        snr = 1
        ylim_max = 6
    elif segmentation == "tumor_s3_snr":
        segmentation = "Unmicst_+_S3"
        snr = 1
        ylim_max = 25
    else:
        segmentation = "Unmicst_+_S3_NoSNR"
        snr = 0
        ylim_max = 10

    case = "_".join(Path(biopsy).stem.split("_")[0:2])
    rounds = rounds[case]

    # plot boxen plot for df, with hue round
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), dpi=200)
    row = 0
    col = 0

    for round in rounds:
        ax = axes[row][col]
        markers = rounds[round]
        df = biopsy_data[markers]
        sns.boxplot(data=df, ax=ax)

        # set title of ax
        ax.set_title(f"Round {round}")
        # set x label of ax
        ax.set_xlabel("Markers")
        # set y label of ax
        ax.set_ylabel("Expression")

        # sety lim
        # ax.set_ylim(0, ylim_max)

        if col == 3:
            row += 1
            col = 0
        else:
            col += 1

    if len(rounds) != 8:
        fig.delaxes(axes[row, col])


    # set figure wide title
    fig.suptitle(
        f"Shared Marker Expression per round\nSegmentation: {segmentation.replace('_', ' ')}\nBiopsy {Path(biopsy).stem.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(
        Path(results_folder, f"{Path(biopsy).stem}_{segmentation}_{snr}_shared_marker_expression_per_round.png"))
