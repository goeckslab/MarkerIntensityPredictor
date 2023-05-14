import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import List
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error


class GCNAutoencoder(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        torch.manual_seed(1234)

        # Encoder layers.
        self.conv_enc1 = GCNConv(num_features, 10)
        self.conv_enc2 = GCNConv(10, 5)
        self.conv_enc3 = GCNConv(5, 2)

        # Decoder layers.
        self.conv_dec1 = GCNConv(2, 5)
        self.conv_dec2 = GCNConv(5, 10)
        self.conv_dec3 = GCNConv(10, num_features)

    def forward(self, x, edge_index):
        # Encode.
        x = self.conv_enc1(x, edge_index)
        x = x.tanh()
        x = self.conv_enc2(x, edge_index)
        x = x.tanh()
        x = self.conv_enc3(x, edge_index)
        x = x.tanh()  # Final embedding.
        embedding = x

        # Decode.
        x = self.conv_dec1(x, edge_index)
        x = x.tanh()
        x = self.conv_dec2(x, edge_index)
        x = x.tanh()
        x = self.conv_dec3(x, edge_index)
        out = x.tanh()

        return out, embedding


def create_results_folder(spatial_radius: str, patient_type: str, replace_mode: str, test_biopsy_name: str) -> [Path,
                                                                                                                int]:
    save_folder = Path(f"gnn/results", patient_type)

    save_folder = Path(save_folder, replace_mode)

    save_folder = Path(save_folder, test_biopsy_name)
    save_folder = Path(save_folder, str(spatial_radius))

    experiment_id = 0

    base_path = Path(save_folder, "experiment_run")
    save_path = Path(str(base_path) + "_" + str(experiment_id))
    while Path(save_path).exists():
        save_path = Path(str(base_path) + "_" + str(experiment_id))
        experiment_id += 1
        print(experiment_id)
        print(save_path)

    created: bool = False
    while not created:
        try:
            save_path.mkdir(parents=True)
            created = True
        except:
            experiment_id += 1
            save_path = Path(str(base_path) + "_" + str(experiment_id))

    return save_path, experiment_id - 1 if experiment_id != 0 else 0


def append_scores_per_iteration(scores: List, test_biopsy_name: str, ground_truth: pd.DataFrame,
                                predictions: pd.DataFrame, hp: bool,
                                type: str, iteration: int, marker: str, spatial_radius: int, experiment_id: int,
                                replace_value: str):
    scores.append({
        "Marker": marker,
        "Biopsy": test_biopsy_name,
        "MAE": mean_absolute_error(predictions[marker], ground_truth[marker]),
        "RMSE": mean_squared_error(predictions[marker], ground_truth[marker], squared=False),
        "HP": int(hp),
        "Type": type,
        "Imputation": 1,
        "Iteration": iteration,
        "FE": spatial_radius,
        "Experiment": experiment_id,
        "Mode": "GNN",
        "Noise": 0,
        "Replace Value": replace_value
    })


def train(data: Data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out, data.x)  # Compute the loss.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rm", "--replace_mode", action="store", type=str, choices=["mean", "zero"], default="zero")
    parser.add_argument("-i", "--iterations", action="store", default=10, type=int)
    parser.add_argument("--mode", action="store", type=str, choices=["ip", "exp"], default="ip")
    parser.add_argument("--spatial", "-sp", action="store", type=int, choices=[23, 46, 92, 138, 184], default=46)
    parser.add_argument("-b", "--biopsy", action="store", type=str, required=True)

    args = parser.parse_args()

    folder = Path("gnn", "data", args.mode)
    spatial_radius = args.spatial
    biopsy_name = args.biopsy
    mode = args.mode
    patient = "_".join(biopsy_name.split("_")[:2])

    if mode == "ip":
        test_set = pd.read_csv(str(Path(folder, biopsy_name, str(spatial_radius), f"test_set.csv")), header=0)
        test_edge_index = torch.load(str(Path(folder, biopsy_name, str(spatial_radius), f"test_edge_index.pt")))
        train_set = pd.read_csv(str(Path(folder, biopsy_name, str(spatial_radius), f"train_set.csv")), header=0)
        train_edge_index = torch.load(str(Path(folder, biopsy_name, str(spatial_radius), f"train_edge_index.pt")))
    else:
        test_set = pd.read_csv(str(Path(folder, f"{biopsy_name}_scaled.csv")),
                               header=0)
        test_edge_index = torch.load(
            str(Path(folder, str(spatial_radius), f"{biopsy_name}_edge_index.pt")))
        train_set = pd.read_csv(str(Path(folder, f"{patient}_excluded_scaled.csv")), header=0)
        train_edge_index = torch.load(
            str(Path("gnn", "data", "exp", str(spatial_radius), f"{patient}_excluded_edge_index.pt")))

    replace_value = args.replace_mode
    iterations = args.iterations

    save_folder, experiment_id = create_results_folder(spatial_radius, mode, replace_value, biopsy_name)

    train_node_features = torch.tensor(train_set.values, dtype=torch.float)
    num_train_cells = train_set.shape[0]
    train_mask = np.full(num_train_cells, False)

    test_node_features = torch.tensor(test_set.values, dtype=torch.float)
    num_test_cells = test_set.shape[0]
    test_mask = np.full(num_test_cells, True)

    train_data = Data(x=train_node_features, train_mask=train_mask, edge_index=train_edge_index,
                      y=train_set['CD45'].values)
    test_data = Data(x=test_node_features, train_mask=test_mask, edge_index=test_edge_index,
                     y=test_set['CD45'].values)

    model = GCNAutoencoder(num_features=len(train_set.columns))  # Create new network.
    criterion = torch.nn.L1Loss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
    out = []

    print("Training model....")
    for epoch in range(100):
        if epoch % 10 == 0:
            print("Epoch: ", epoch)
        loss, h = train(train_data)

    # Evaluate model performance on test set.
    model.eval()

    scores = []
    predictions = {}
    # prepare predictions dict
    for i in range(iterations):
        predictions[i] = pd.DataFrame(columns=test_set.columns)

    for marker in test_set.columns:
        input_data = pd.DataFrame(test_data.x, columns=test_set.columns).copy()
        if replace_value == "zero":
            input_data[marker] = 0
        elif replace_value == "mean":
            input_data[marker] = input_data[marker].mean()

        marker_prediction = input_data.copy()

        for i in tqdm(range(iterations)):
            with torch.no_grad():
                # convert data to tensor
                marker_prediction = torch.tensor(marker_prediction.values, dtype=torch.float)
                predicted_intensities, h = model(marker_prediction, test_data.edge_index)
                predicted_intensities = pd.DataFrame(data=predicted_intensities, columns=test_set.columns)
                predictions[i][marker] = predicted_intensities[marker].values

                # Extract the reconstructed marker
                imputed_marker = predicted_intensities[marker].values
                # copy the original dataset and replace the marker in question with the imputed data
                marker_prediction = input_data.copy()
                marker_prediction[marker] = imputed_marker

                append_scores_per_iteration(scores=scores, test_biopsy_name=biopsy_name, predictions=marker_prediction,
                                            ground_truth=test_set,
                                            hp=False, type=mode, iteration=i, marker=marker,
                                            spatial_radius=spatial_radius, experiment_id=experiment_id,
                                            replace_value=replace_value)

    # Convert to df
    scores = pd.DataFrame(scores)

    print(predictions.keys())

    # Save data
    scores.to_csv(f"{save_folder}/scores.csv", index=False)
    for key, value in predictions.items():
        value.to_csv(f"{save_folder}/{key}_predictions.csv",
                     index=False)
