import argparse
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    results_folder = Path("results")

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--files", type=argparse.FileType('r'), required=False, action="store",
                                help="The files used to generate the clusters", nargs='+')

    args =  parser.parse_args()
    frames = []
    for file in args.files:
        data = pd.read_csv(file)
        frames.append(data)

    df = pd.concat(frames).reset_index()
    df.to_csv(f"{results_folder}/merged_dataset.csv", index=False)
