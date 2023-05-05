import argparse, os
import sys
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    print("Combining datasets...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory to load data from", required=True)
    parser.add_argument("--target", help="The target biopsy", required=True, type=str)
    parser.add_argument("--output_dir", help="Directory to save data to", required=True)
    args = parser.parse_args()
    target = args.target  # The target biopsy which should be excluded
    directory = args.dir

    if Path(f"{args.output_dir}/{target}_excluded_dataset.csv").exists():
        sys.exit(0)

    train = []
    patient = "_".join(Path(args.target).stem.split("_")[:2])
    for root, dirs, files in os.walk(directory):
        for name in files:
            if Path(name).suffix == ".csv" and patient not in name and "excluded" not in name:
                print("Loading {}".format(name))
                train.append(pd.read_csv(os.path.join(root, name), header=0))

    train = pd.concat(train, axis=0)
    train.to_csv(f"{args.output_dir}/{target}_excluded_dataset.csv", index=False)
