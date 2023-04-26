import argparse, os
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory to load data from", required=True)
    parser.add_argument("--target", help="The target biopsy", required=True)
    parser.add_argument("--output_dir", help="Directory to save data to", required=True)
    parser.add_argument("-exp", "--exclude_patient", action="store_true", default=False,
                        help="Exclude the target patient from being loaded")
    args = parser.parse_args()
    target = args.target  # The target biopsy which should be excluded
    directory = args.dir
    exclude_patient = args.exclude_patient
    patient = "_".join(Path(args.target).stem.split("_")[:2])

    if exclude_patient:
        print("Excluding patient from dataset...")

    if exclude_patient:
        output_path = Path(f"{args.output_dir}/{patient}_excluded_dataset.csv")
    else:
        output_path = Path(f"{args.output_dir}/{target}_excluded_dataset.csv")

    if output_path.exists():
        print("File already exists. Skipping...")
        exit(0)

    train = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if args.exclude_patient:
                if Path(name).suffix == ".csv" and patient not in name and "excluded" not in name:
                    print("Loading {}".format(name))
                    train.append(pd.read_csv(os.path.join(root, name), header=0))
            else:
                if Path(name).suffix == ".csv" and args.target not in name and "excluded" not in name:
                    print("Loading {}".format(name))
                    train.append(pd.read_csv(os.path.join(root, name), header=0))

    train = pd.concat(train, axis=0)

    if exclude_patient:
        train.to_csv(f"{args.output_dir}/{patient}_excluded_dataset.csv", index=False)
    else:
        train.to_csv(f"{args.output_dir}/{target}_excluded_dataset.csv", index=False)
