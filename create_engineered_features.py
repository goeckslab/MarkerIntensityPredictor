from sklearn.neighbors import KDTree
import argparse
from pathlib import Path
import pandas as pd

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
NON_MARKERS = ['CellID', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
               'Solidity', 'Extent', 'Orientation']
SPATIAL_COORDS = ['X_centroid', 'Y_centroid']

TO_REMOVE = ["DNA1", "DNA2", "DNA3", "DNA4", "DNA5", "DNA6", "DNA7", "DNA8", "DNA9", "DNA10", "DNA11", "DNA12", "DNA13",
             "pERK-555", "goat-anti-rabbit", "A555", "donkey-anti-mouse"]


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training only if folder arg is provided.")

    parser.add_argument("--mode", "-m", action="store", required=True, choices=["sp", "bio"],
                        help="The mode of finding the nearest neighbors")

    parser.add_argument("--neighbors", "-n", action="store", required=False, default=6, type=int,
                        help="The amount of neighbors")

    parser.add_argument("--radius", "-r", action="store", required=False, default=46, type=int,
                        help="The radius around the cell in px")  # 2 cells

    parser.add_argument("--output", "-o", action="store", required=True)

    return parser.parse_args()


def calc_mean(features, feature, neighbors, index):
    cell_neighbors = neighbors[index][1:]
    return features.iloc[cell_neighbors][feature].mean()


if __name__ == '__main__':
    args = get_args()

    neighbor_count = args.neighbors
    mode = args.mode
    radius = args.radius

    results_folder = args.output
    if mode == "sp":
        results_folder = Path(f"{results_folder}_sp_{str(radius)}")
    else:
        results_folder = Path(f"{results_folder}_bio_{str(neighbor_count)}")

    if not results_folder.exists():
        results_folder.mkdir(parents=True, exist_ok=True)

    file_name: str = Path(args.file).stem

    biopsy: pd.DataFrame = pd.read_csv(args.file, delimiter=",", header=0)

    print(f"Processing file: {file_name}")
    print(f"Mode: {mode}")

    for column in TO_REMOVE:
        if column in biopsy.columns:
            biopsy.drop(columns=[column], inplace=True)

    if mode == "sp":
        kdt = KDTree(biopsy[SPATIAL_COORDS], leaf_size=30, metric='euclidean')
        neighbors = kdt.query_radius(biopsy[['X_centroid', 'Y_centroid']], r=radius, return_distance=False)
    elif mode == "biomorph":
        sub_df = biopsy.drop(columns=[SPATIAL_COORDS], inplace=True)
        sub_df = sub_df.drop(columns=['CellID'], inplace=True)
        kdt = KDTree(sub_df, leaf_size=30, metric='euclidean')
        neighbors = kdt.query(sub_df, k=neighbor_count, return_distance=False)
    else:
        sub_df = biopsy[SHARED_MARKERS].copy()
        kdt = KDTree(sub_df, leaf_size=30, metric='euclidean')
        neighbors = kdt.query(sub_df, k=neighbor_count, return_distance=False)

    biopsy = biopsy[SHARED_MARKERS].copy()
    for feature in biopsy.columns:
        biopsy[f"{feature}_mean"] = biopsy.apply(
            lambda row: calc_mean(biopsy, feature, neighbors, row.name), axis=1)

    # Create results path
    results_folder.mkdir(parents=True, exist_ok=True)

    biopsy.fillna(0, inplace=True)
    biopsy.to_csv(Path(results_folder, f"{file_name}.csv"), index=False)
