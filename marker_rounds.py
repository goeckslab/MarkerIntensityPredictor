import argparse, math, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def load_mae_scores():
    biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
    # load mesmer mae scores from data mesmer folder and all subfolders

    mae_scores = []
    for root, dirs, files in os.walk("mesmer"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_non_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(mae_scores) == 48, "There should be 48 mae scores files"
    mae_scores = pd.concat(mae_scores, axis=0).reset_index(drop=True)

    return mae_scores


if __name__ == '__main__':
    # Load data
    r_9_2 = pd.read_csv("data/rounds/CycIF_HTAN9_2_Tumor_markers.csv")
    r_9_3 = pd.read_csv("data/rounds/CycIF_HTAN9_3_Tumor_markers.csv")
    r_9_14 = pd.read_csv("data/rounds/CycIF_HTAN9_14_Tumor_markers.csv")
    r_9_15 = pd.read_csv("data/rounds/CycIF_HTAN9_15_Tumor_markers.csv")

    # create a new columns round based on channel_number modulo 3
    r_9_2["Round"] = [channel_number // 4 for channel_number in list(r_9_2.index)]
    r_9_2["Patient"] = "9 2"

    r_9_3["Round"] = [channel_number // 4 for channel_number in list(r_9_3.index)]
    r_9_3["Patient"] = "9 3"

    r_9_14["Round"] = [channel_number // 4 for channel_number in list(r_9_14.index)]
    r_9_14["Patient"] = "9 14"

    r_9_15["Round"] = [channel_number // 4 for channel_number in list(r_9_15.index)]
    r_9_15["Patient"] = "9 15"

    # merge all patient together
    rounds = pd.concat([r_9_2, r_9_3, r_9_14, r_9_15], axis=0).reset_index(drop=True)

    # print all rounds of ck19

    mae_scores = load_mae_scores()
    # create new column patient based on biopsy value
    mae_scores["Patient"] = [" ".join(biopsy.split("_")[:2]) for biopsy in mae_scores["Biopsy"]]

    # merge mae scores with rounds
    rounds = rounds.merge(mae_scores, on=["Patient"])
    # select shared markers
    rounds = rounds[rounds["Marker"].isin(SHARED_MARKERS)]
    print(rounds)
    rounds.to_csv("rounds.csv", index=False)

    #sns.barplot(x="Biopsy", y="Score", hue="Round", data=rounds)
    #plt.show()

    sns.catplot(
        data=rounds, x="Biopsy", y="Score", col="marker_name",
        kind="bar", height=4, aspect=.6,
    )
    plt.show()
    # calculate difference between round 5 and 7 for ck19
