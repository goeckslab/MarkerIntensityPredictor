import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from pathlib import Path

if __name__ == '__main__':

    # load mesmer mae scores from data mesmer folder and all subfolders

    scores = []
    for root, dirs, files in os.walk("data/scores"):
        for name in files:
            if Path(name).suffix == ".csv" and "_EN_" in name:
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 48, "Not all biopsies have been processed"
