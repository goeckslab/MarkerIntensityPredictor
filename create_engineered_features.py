from sklearn.neighbors import KDTree
import argparse
from pathlib import Path
import pandas as pd

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
NON_MARKERS = ['CellID', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
               'Solidity', 'Extent', 'Orientation']
SPATIAL_COORDS = ['X_centroid', 'Y_centroid']


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", action="store", required=True,
                        help="The file to use for imputation. Will be excluded from training only if folder arg is provided.")
    parser.add_argument("--sub", action="store", required=False, default=None,
                        help="Select a given subset range of cells")

    parser.add_argument("--mode", "-m", action="store", required=True, choices=["sp", "bio", "biomorph"],
                        help="The mode of finding the nearest neighbors")

    parser.add_argument("--neighbors", "-n", action="store", required=False, default=6, type=int,
                        help="The amount of neighbors")

    parser.add_argument("--radius", "-r", action="store", required=False, default=46, type=int,
                        help="The radius around the cell in px")

    return parser.parse_args()


def calc_mean(features, feature, neighbors, index):
    cell_neighbors = neighbors[index][1:]
    return features.iloc[cell_neighbors][feature].mean()


if __name__ == '__main__':
    args = get_args()

    neighbor_count = args.neighbors
    radius = args.radius

    if args.sub is not None:
        sub_count = int(args.sub)

    mode = args.mode
    results_folder = Path("data/feature_engineered")

    file_name: str = Path(args.file).stem

    biopsy: pd.DataFrame = pd.read_csv(args.file, delimiter=",", header=0)

    print(f"Processing file: {file_name}")
    print(f"Mode: {mode}")

    if mode == "sp":
        biopsy = biopsy.drop(columns=[NON_MARKERS], inplace=True)
        kdt = KDTree(biopsy[SPATIAL_COORDS], leaf_size=30, metric='euclidean')
        neighbors = kdt.query_radius(biopsy[['X_centroid', 'Y_centroid']], r=radius, return_distance=False)
        results_folder = Path(results_folder, "sp")
    elif mode == "biomorph":
        sub_df = biopsy.drop(columns=[SPATIAL_COORDS], inplace=True)
        sub_df = sub_df.drop(columns=['CellID'], inplace=True)
        kdt = KDTree(sub_df, leaf_size=30, metric='euclidean')
        neighbors = kdt.query(sub_df, k=neighbor_count, return_distance=False)
        results_folder = Path(results_folder, "biomorph")
    else:
        sub_df = biopsy.drop(columns=[NON_MARKERS])
        sub_df = sub_df.drop(columns=[SPATIAL_COORDS])
        kdt = KDTree(sub_df, leaf_size=30, metric='euclidean')
        neighbors = kdt.query(sub_df, k=neighbor_count, return_distance=False)
        results_folder = Path(results_folder, "bio")

    excluded_features = NON_MARKERS + SPATIAL_COORDS

    for feature in biopsy.features:
        if feature in excluded_features:
            continue

        biopsy.features[f"{feature}_mean"] = biopsy.features.apply(
            lambda row: calc_mean(biopsy.features, feature, neighbors, row.name), axis=1)

    df = biopsy.features.copy()
    df = df.join(pd.DataFrame(biopsy.cells_ids))
    df.rename(columns={0: "CellID"}, inplace=True)

    if args.sub is not None:
        results_folder = Path(results_folder, args.sub)

    if mode == "sp":
        results_folder = Path(results_folder, str(radius))
    else:
        results_folder = Path(results_folder, str(neighbor_count))

    # Create results path
    results_folder.mkdir(parents=True, exist_ok=True)

    df.fillna(0, inplace=True)
    df.to_csv(Path(results_folder, f"{file_name}.csv"), index=False)
