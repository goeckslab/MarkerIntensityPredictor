import argparse, math, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

save_folder = Path("data", "cleaned_data", "rounds")

if __name__ == '__main__':

    # create save folder if not exists
    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

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

    rounds.to_csv("rounds.csv", index=False)
    # remove AF_Channel column and channel_number from rounds
    rounds = rounds.drop(columns=["AF_channel", "channel_number"], axis=1)
    # rename marker_name to Marker
    rounds = rounds.rename(columns={"marker_name": "Marker"})
    print(rounds)
    # save to save path
    rounds.to_csv(Path(save_folder, "rounds.csv"), index=False)