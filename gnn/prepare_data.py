import argparse, os, sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import torch
from sklearn.neighbors import BallTree
from time import sleep

save_path = Path("gnn/data")
markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']


def create_save_folder(save_path: Path, biopsy_name: str, mode: str, spatial: str) -> Path:
    save_path = Path(save_path, mode)
    if mode == "ip":
        save_path = Path(save_path, biopsy_name)
    save_path = Path(save_path, spatial)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    return save_path


def prepare_data_for_exp_gnn(biopsy_name: str, dataset: pd.DataFrame, spatial: int):
    print(f"Preparing {biopsy_name}...")
    marker_df = dataset[markers]
    marker_df = np.log10(marker_df + 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    marker_df = pd.DataFrame(scaler.fit_transform(marker_df),
                             columns=marker_df.columns)

    print("Calculating distances...")
    exp_tree = BallTree(dataset[['X_centroid', 'Y_centroid']], leaf_size=2)

    ids = exp_tree.query_radius(dataset[['X_centroid', 'Y_centroid']], r=spatial)

    # convert indexes to a list of lists
    edge_index = []
    for i in range(len(ids)):
        for j in range(len(ids[i])):
            edge_index.append([i, ids[i][j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    marker_df.to_csv(str(Path("gnn", "data", "exp", f"{biopsy_name}_scaled.csv")), index=False)
    torch.save(edge_index, str(Path("gnn", "data", "exp", str(spatial), f"{biopsy_name}_edge_index.pt")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The test biopsy", required=True)
    parser.add_argument("--mode", choices=["ip", "exp"], default="ip")
    parser.add_argument("-sp", "--spatial", choices=[23, 46, 92, 138, 184],
                        help="The distance in pixels between nodes", default=46, type=int)

    args = parser.parse_args()

    biopsy = args.biopsy
    biopsy_name = Path(biopsy).stem
    patient = "_".join(Path(args.biopsy).stem.split("_")[:2])
    mode = args.mode
    spatial = args.spatial

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

        if Path(save_path, f"test_edge_index.pt").exists() and Path(save_path, "test_set.csv").exists() and Path(
                save_path, "train_set.csv").exists() and Path(save_path, "train_edge_index.pt").exists():
            print("Data already exists. Skipping...")
            sys.exit(0)

        # load train set from the folder of the biopsy
        train_set = pd.read_csv(str(Path("data/tumor_mesmer", f"{train_biopsy}.csv")), header=0)

        # Transform the train data.
        train_markers_df = train_set[markers]
        train_markers_df = np.log10(train_markers_df + 1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        train_markers_df = pd.DataFrame(min_max_scaler.fit_transform(train_markers_df),
                                        columns=train_markers_df.columns)

        # Transform the test data.
        test_markers_df = test_set[markers]
        test_markers_df = np.log10(test_markers_df + 1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        test_markers_df = pd.DataFrame(min_max_scaler.fit_transform(test_markers_df), columns=test_markers_df.columns)

        print("Creating train edge index...")

        # set distance threshold and create edge index
        threshold = spatial
        print("Calculating distances...")
        tree = BallTree(train_set[['X_centroid', 'Y_centroid']], leaf_size=2)
        ind = tree.query_radius(train_set[['X_centroid', 'Y_centroid']], r=spatial)
        # convert indexes to a list of lists
        edge_index = []
        for i in range(len(ind)):
            for j in range(len(ind[i])):
                edge_index.append([i, ind[i][j]])

        train_edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        print("Creating test edge index...")
        tree = BallTree(test_set[['X_centroid', 'Y_centroid']], leaf_size=2)
        ind = tree.query_radius(test_set[['X_centroid', 'Y_centroid']], r=spatial)
        # convert indexes to a list of lists
        edge_index = []
        for i in range(len(ind)):
            for j in range(len(ind[i])):
                edge_index.append([i, ind[i][j]])

        test_edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        print(save_path)
        print(biopsy_name)

        # save the data
        torch.save(train_edge_index, str(Path(str(save_path), f"train_edge_index.pt")))
        torch.save(test_edge_index, str(Path(str(save_path), f"test_edge_index.pt")))
        train_markers_df.to_csv(str(Path(str(save_path), f"train_set.csv")), index=False)
        test_markers_df.to_csv(str(Path(str(save_path), f"test_set.csv")), index=False)

    elif mode == "exp":
        train_set = {}
        test_set = pd.DataFrame()
        for root, dirs, files in os.walk(str(Path("data/tumor_mesmer"))):
            for name in files:
                if Path(name).suffix == ".csv" and patient not in name and "excluded" not in name:
                    print("Loading {}".format(Path(name).stem))
                    train_set[Path(name).stem] = pd.read_csv(os.path.join(root, name), header=0)

                if Path(name).suffix == ".csv" and Path(name).stem == biopsy_name:
                    print("Loading test biopsy {}".format(Path(name).stem))
                    test_set = pd.read_csv(os.path.join(root, name), header=0)

        assert len(train_set) == 6, "Expected 6 files, got {}".format(len(train_set))

        for train_biopsy_name, dataset in train_set.items():
            # check if edge_index.pt does exist in folder
            if not Path("gnn", "data", "exp", str(spatial), f"{train_biopsy_name}_edge_index.pt").exists() or \
                    not (Path("gnn", "data", "exp", f"{train_biopsy_name}_scaled.csv").exists()):
                prepare_data_for_exp_gnn(train_biopsy_name, dataset, spatial)

        # prepare test set
        if not Path("gnn", "data", "exp", str(spatial), f"{biopsy_name}_edge_index.pt").exists() or \
                not (Path("gnn", "data", "exp", f"{biopsy_name}_scaled.csv").exists()):
            prepare_data_for_exp_gnn(biopsy_name, test_set, spatial)

        # Potential exacloud fix
        sleep(20)

        # load indexes
        indexes = []
        dfs = []
        max_id = 0
        for train_biopsy_name in train_set.keys():
            # load scaled data set
            current_df = pd.read_csv(str(Path("gnn", "data", "exp", f"{train_biopsy_name}_scaled.csv")), header=0)

            # load index
            node_index = torch.load(str(Path("gnn", "data", "exp", str(spatial), f"{train_biopsy_name}_edge_index.pt")))

            # update all values of the node index array with the index value
            node_index = node_index + max_id

            indexes.append(node_index)
            dfs.append(current_df)

            # update index
            max_id = max_id + len(current_df)

        patient_excluded_df = pd.concat(dfs, ignore_index=True)
        patient_excluded_edges = torch.cat(indexes, dim=1)

        # save patient excluded edges
        torch.save(patient_excluded_edges,
                   str(Path(save_path, f"{patient}_excluded_edge_index.pt")))

        # save patient excluded df
        patient_excluded_df.to_csv(
            str(Path("gnn", "data", "exp", f"{patient}_excluded_scaled.csv")), index=False)


    else:
        raise ValueError("Invalid mode")
