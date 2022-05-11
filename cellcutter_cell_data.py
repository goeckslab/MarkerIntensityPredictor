import pandas as pd
import argparse


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file used to extract the required data for cellcutter")
    parser.add_argument("--out", "-o", action="store", required=True, help="The file name output")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    data = pd.read_csv(args.file)

    data = data[["CellID", "X_centroid", "Y_centroid"]]
    data.to_csv(args.out)
