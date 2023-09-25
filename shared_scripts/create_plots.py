import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", help="Ground truth values")
    parser.add_argument("--predicted", help="predicted values")
    parser.add_argument("--marker", help="the predicted marker")
    parser.add_argument("--output_dir", help="Output directory")
    args = parser.parse_args()
    marker = args.marker

    truth = pd.read_csv(args.truth, delimiter="\t", header=0)
    predicted = pd.read_csv(args.predicted, delimiter=",", header=None)
    predicted.rename(columns={0: marker}, inplace=True)

    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(truth[[marker]], predicted[[marker]], alpha=0.5, label=marker)
    plt.plot(np.unique(truth[[marker]].values.flatten()),
             np.poly1d(np.polyfit(truth[[marker]].values.flatten(), predicted[[marker]].values.flatten(), 1))(
                 np.unique(truth[[marker]].values.flatten())), color='red')

    tested = Path(args.truth).stem
    test_splits = tested.split("_")
    if test_splits[2] == 2:
        tested = " ".join(test_splits[:2]) + " 1"
        trained = " ".join(test_splits[:2]) + " 2"
    else:
        tested = " ".join(test_splits[:2]) + " 2"
        trained = " ".join(test_splits[:2]) + " 1"

    plt.legend(loc='lower right')
    plt.ylabel("Test Cell")
    plt.xlabel("Predicted Cell")
    plt.suptitle(f"Expression of {marker}", x=0.58)
    plt.title(f"Predicted vs. Test cell. \n Trained on {trained}. Tested on {tested}")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{marker}.png")
