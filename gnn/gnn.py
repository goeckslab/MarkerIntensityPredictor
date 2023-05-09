import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, argparse
from pathlib import Path


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

def create_results_folder(spatial_radius: str) -> Path:
    if replace_all_markers:
        save_folder = Path(f"ae_imputation", f"{patient_type}_replace_all")
    else:
        save_folder = Path(f"ae_imputation", patient_type)

    save_folder = Path(save_folder, replace_mode)
    if add_noise:
        save_folder = Path(save_folder, "noise")
    else:
        save_folder = Path(save_folder, "no_noise")

    save_folder = Path(save_folder, test_biopsy_name)
    save_folder = Path(save_folder, spatial_radius)

    suffix = 1

    base_path = Path(save_folder, "experiment_run")
    save_path = Path(str(base_path) + "_0")
    while Path(save_path).exists():
        save_path = Path(str(base_path) + "_" + str(suffix))
        suffix += 1

    created: bool = False
    if not save_path.exists():
        while not created:
            try:
                save_path.mkdir(parents=True)
                created = True
            except:
                suffix += 1
                save_path = Path(str(base_path) + "_" + str(suffix))

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder",
                        help="The folder where all files are stored. Should not be a top level folder ", required=True)

    args = parser.parse_args()

    folder = args.folder
    biopsy = Path(folder).stem
    patient = "_".join(Path(folder).stem.split("_")[:2])
    test_set = pd.read_csv(str(Path(folder, f"test_set.csv")), header=0)
    test_edge_index = torch.load(str(Path(folder, f"test_edge_index.pt")))
    train_set = pd.read_csv(str(Path(folder, f"train_set.csv")), header=0)
    train_edge_index = torch.load(str(Path(folder, f"train_edge_index.pt")))

    print(biopsy)
    print(folder)
    print(patient)

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


    def train(data: Data):
        optimizer.zero_grad()  # Clear gradients.
        out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out, data.x)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss, h


    for epoch in range(100):
        loss, h = train(train_data)

    # Evaluate model performance on test set.
    model.eval()
    with torch.no_grad():
        out, h = model(test_data.x, test_data.edge_index)
        loss = criterion(out, test_data.x)  # Compute the loss.
    print("total loss", loss.item())

    print("per marker loss:")
    out_test = out
    data_test = test_data.x
    for i, col in enumerate(test_set.columns):
        out_marker = out_test[:, i]
        data_marker = data_test[:, i]
        loss = criterion(out_marker, data_marker)
        print("\t", col, loss.item())
