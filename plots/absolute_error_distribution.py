import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--errors", help="where to find the error scores")
    parser.add_argument("-m", "--markers", nargs='+')
    args = parser.parse_args()

    # load error scores
    errors = pd.read_csv(args.errors)

    # select only marker in args.markers
    if args.markers:
        errors = errors[args.markers]

    title = ""
    if "in_patient" in str(args.errors):
        title = "In patient"
    else:
        title = "Out patient"

    biopsy_name = Path(args.errors).stem.replace("_", " ")

    title = f"Biopsy {biopsy_name}\nLudwig {title}\nDistribution of absolute error per cell"

    # create violinplot for all markers

    fig = plt.figure(dpi=200, figsize=(10, 5))
    sns.violinplot(data=errors, inner='box')
    plt.title(title)
    plt.ylabel("Absolute error")
    plt.xlabel("Marker")
    plt.tight_layout()
    plt.savefig(f"{Path(args.errors).parent}/{biopsy_name.replace(' ', '_')}.png")
