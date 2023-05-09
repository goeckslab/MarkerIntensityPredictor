import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import torch

save_path = Path("gnn/data")
markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']


def create_save_folder(save_path: Path, biopsy, mode: str, spatial: str) -> Path:
    save_path = Path(save_path, mode)
    save_path = Path(save_path, biopsy)
    save_path = Path(save_path, spatial)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The test biopsy", required=True)
    parser.add_argument("-m", "--mode", choices=["ip", "exp"], default="ip")
    parser.add_argument("-sp", "--sp", choices=[23, 46, 92, 138, 184],
                        help="The distance in pixels between nodes", default=46, type=int)

    args = parser.parse_args()

    biopsy = args.biopsy
    biopsy_name = Path(biopsy).stem
    patient = "_".join(Path(args.biopsy).stem.split("_")[:2])
    mode = args.mode
    spatial = args.sp

    print(f"Mode: {mode}")
    print(f"Biopsy: {biopsy_name}")
    print(f"Spatial: {spatial}")

    test_set = pd.read_csv(str(Path("data/tumor_mesmer", f"{biopsy_name}.csv")), header=0)

    save_path = create_save_folder(save_path, biopsy_name, mode, str(spatial))

    if mode == "ip":
        if biopsy_name[-1] == "1":
            train_biopsy = biopsy_name[:-1] + "2"
        else:
            train_biopsy = biopsy_name[:-1] + "1"

        print(f"Train Biopsy: {train_biopsy}")

        # load train set from the folder of the biopsy
        train_set = pd.read_csv(str(Path("data/tumor_mesmer", f"{train_biopsy}.csv")), header=0)

    elif mode == "exp":
        train_set = []
        for root, dirs, files in os.walk(str(Path("data/tumor_mesmer"))):
            for name in files:
                if Path(name).suffix == ".csv" and patient not in name and "excluded" not in name:
                    print("Loading {}".format(name))
                    train_set.append(pd.read_csv(os.path.join(root, name), header=0))

        assert len(train_set) == 6, "Expected 6 files, got {}".format(len(train_set))

        train_set = pd.concat(train_set, axis=0)

    else:
        raise ValueError("Invalid mode")



    # Transform the train data.
    train_markers_df = train_set[markers]
    train_markers_df = np.log10(train_markers_df + 1)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_markers_df = pd.DataFrame(min_max_scaler.fit_transform(train_markers_df), columns=train_markers_df.columns)

    # Transform the test data.
    test_markers_df = test_set[markers]
    test_markers_df = np.log10(test_markers_df + 1)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    test_markers_df = pd.DataFrame(min_max_scaler.fit_transform(test_markers_df), columns=test_markers_df.columns)

    # extract x and y coordinates as numpy array
    train_coords = train_set[['X_centroid', 'Y_centroid']].values
    test_coords = train_set[['X_centroid', 'Y_centroid']].values

    print("Calculating pairwise distances for train set...")
    # compute pairwise distances between nodes
    train_distances = cdist(train_coords, train_coords)
    print("Calculating pairwise distances for test set...")
    test_distances = cdist(test_coords, test_coords)

    print("Creating train edge index...")

    # set distance threshold and create edge index
    threshold = spatial
    train_edge_index = torch.tensor(
        [(i, j) for i in range(len(train_markers_df)) for j in range(len(train_markers_df)) if
         i != j and train_distances[i, j] < threshold],
        dtype=torch.long).t()

    print("Creating test edge index...")
    test_edge_index = torch.tensor(
        [(i, j) for i in range(len(test_markers_df)) for j in range(len(test_markers_df)) if
         i != j and test_distances[i, j] < threshold],
        dtype=torch.long).t()

    print(save_path)
    print(biopsy_name)

    # save the data
    torch.save(train_edge_index, str(Path(str(save_path), f"train_edge_index.pt")))
    torch.save(test_edge_index, str(Path(str(save_path), f"test_edge_index.pt")))
    train_markers_df.to_csv(str(Path(str(save_path), f"train_set.csv")), index=False)
    test_markers_df.to_csv(str(Path(str(save_path), f"test_set.csv")), index=False)
