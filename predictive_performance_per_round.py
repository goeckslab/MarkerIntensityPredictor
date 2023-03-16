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

results_folder = Path("results/scores_by_round")

if __name__ == '__main__':
    if not results_folder.exists():
        results_folder.mkdir(parents=True)

    # Check whether marker expression is high

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", help='Input file')
    parser.add_argument("-hyper", "--hyper", action="store_true", default=False)
    parser.add_argument("-f", "--fe", action="store_true", default=False)
    parser.add_argument("--snr", action="store_true", default=True)
    args = parser.parse_args()

    biopsy = args.biopsy
    segmentation = biopsy.split("/")[1]
    print(segmentation)
    if segmentation == "tumor_mesmer":
        segmentation = "Mesmer"
    elif segmentation == "tumor_s3_snr":
        segmentation = "Unmicst_+_S3"

    hyper = 0 if not args.hyper else 1
    fe = None if not args.fe else "SP"

    # Load score data from data scores folder

    scores = []
    for model in ["EN", "Ludwig"]:
        snr = 0 if not args.snr else 1
        scores.append(pd.read_csv(Path("data/scores",
                                       f"{Path(args.biopsy).stem}_IP_{'_'.join(segmentation.split(' '))}_{snr}_{fe}_{model}_{hyper}_scores.csv")))
        scores.append(pd.read_csv(Path("data/scores",
                                       f"{Path(args.biopsy).stem}_OP_{'_'.join(segmentation.split(' '))}_{snr}_{fe}_{model}_{hyper}_scores.csv")))

    assert len(scores) == 4, "Not all scores are loaded"
    scores = pd.concat(scores)
    # create new column which merges type and model together
    scores["Type"] = scores["Type"] + "_" + scores["Mode"]
    # replace _ with spaces for Type column
    scores["Type"] = scores["Type"].str.replace("_", " ")
    case = "_".join(Path(biopsy).stem.split("_")[0:2])
    rounds = rounds[case]

    # plot boxen plot for each round

    # plot boxen plot for df, with hue round
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), dpi=200)
    row = 0
    col = 0

    score = "MAE"
    for round in rounds:
        ax = axes[row][col]
        markers = rounds[round]
        # Select all markers for round and mae scores
        df = scores[scores["Marker"].isin(markers)][["Marker", score, "Type"]]

        sns.barplot(data=df, x="Marker", y="MAE", hue="Type", ax=ax)

        ax.legend_ = None
        # set title of ax
        ax.set_title(f"Round {round}")
        # set x label of ax
        ax.set_xlabel("Markers")
        # set y label of ax
        ax.set_ylabel(score)
        # set ylim of ax
        ax.set_ylim(0, 0.55)

        if col == 3:
            row += 1
            col = 0
        else:
            col += 1

    if len(rounds) != 8:
        fig.delaxes(axes[row, col])

    # set figure wide title
    fig.suptitle(
        f"Shared Marker MAE per round\n{segmentation.replace('_', ' ')} {'SNR' if snr else ''} \nBiopsy {Path(biopsy).stem.replace('_', ' ')}")

    # set bboxes of legend outside of figure
    # use axes legend attributes to create a fig legen
    fig.legend(*ax.get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(1.001, 0.2))
    plt.tight_layout()

    plt.savefig(
        Path(results_folder, f"{Path(biopsy).stem}_{segmentation}_{snr}_marker_prediction_per_round.png"))
