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

if __name__ == '__main__':

    # Check whether marker expression is high

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", help='Input file')
    args = parser.parse_args()

    biopsy = args.biopsy
    biopsy_data = pd.read_csv(biopsy)

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

        if col == 3:
            row += 1
            col = 0
        else:
            col += 1

    # set figure wide title
    fig.suptitle(f"Shared Marker Expression per round\nBiopsy {Path(biopsy).stem.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f"{Path(biopsy).stem}_shared_marker_expression_per_round.png")
