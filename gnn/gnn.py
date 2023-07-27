import random
from sklearn.neighbors import BallTree
import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import List, Dict
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, argparse, logging
from pathlib import Path
from tqdm import tqdm
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ae_imputation_m/debug.log"),
                        logging.StreamHandler()
                    ])


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "debug.log")
    file_logger = logging.FileHandler(save_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for handler in log.handlers[:]:  # remove all old handlers
        log.removeHandler(handler)
    log.addHandler(file_logger)
    log.addHandler(logging.StreamHandler())


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


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
                                replace_value: str, subset: int):
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
        "Experiment": int(f"{experiment_id}{subset}"),
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


def save_scores(save_folder: str, file_name: str, scores: List[Dict]):
    if Path(f"{save_folder}/{file_name}").exists():
        temp_df = pd.read_csv(f"{save_folder}/{file_name}")
        scores = pd.concat([temp_df, pd.DataFrame(scores)])
    pd.DataFrame(scores).to_csv(f"{save_folder}/{file_name}", index=False)


def impute_marker(test_data: Data, subset: int, scores: List, all_predictions: Dict, file_name: str, save_folder: str,
                  store_predictions: bool, columns: List, replace_value: str, iterations: int, biopsy_name: str):
    try:
        for marker in columns:
            input_data = pd.DataFrame(test_data.x, columns=columns).copy()
            if replace_value == "zero":
                input_data[marker] = 0
            elif replace_value == "mean":
                input_data[marker] = input_data[marker].mean()

            marker_prediction = input_data.copy()

            for iteration in tqdm(range(iterations)):
                with torch.no_grad():
                    # convert data to tensor
                    marker_prediction = torch.tensor(marker_prediction.values, dtype=torch.float)
                    predicted_intensities, h = model(marker_prediction, test_data.edge_index)
                    predicted_intensities = pd.DataFrame(data=predicted_intensities, columns=columns)

                    if store_predictions:
                        all_predictions[iteration][marker] = predicted_intensities[marker].values

                    # Extract the reconstructed marker
                    imputed_marker = predicted_intensities[marker].values
                    # copy the original dataset and replace the marker in question with the imputed data
                    marker_prediction = input_data.copy()
                    marker_prediction[marker] = imputed_marker

                    append_scores_per_iteration(scores=scores, test_biopsy_name=biopsy_name,
                                                predictions=marker_prediction,
                                                ground_truth=pd.DataFrame(test_data.x, columns=columns),
                                                hp=False, type=mode, iteration=iteration, marker=marker,
                                                spatial_radius=spatial_radius, experiment_id=experiment_id,
                                                replace_value=replace_value, subset=subset)

                    logging.debug(f"Finished iteration {iteration} for marker {marker}.")
                    if iteration % 20 == 0:
                        logging.debug("Performing temp save...")
                        save_scores(save_folder=save_folder, file_name=file_name, scores=scores)
                        scores = []

        if len(scores) > 0:
            save_scores(save_folder=save_folder, file_name=file_name, scores=scores)
            scores = []

    except KeyboardInterrupt as ex:
        logging.error("Keyboard interrupt detected.")
        logging.error("Saving scores...")
        if len(scores) > 0:
            save_scores(scores=scores, save_folder=save_folder, file_name=file_name)
        raise
    except BaseException as ex:
        logging.error(ex)
        logging.error("Test truth:")
        logging.error(test_data)
        logging.error("Predictions:")
        logging.error(predictions)
        logging.error(mode)
        logging.error(spatial_radius)
        logging.error(experiment_id)
        logging.error(replace_value)
        raise


def calculate_edge_indexes(dataset: pd.DataFrame, spatial: int) -> torch.Tensor:
    exp_tree = BallTree(dataset[['X_centroid', 'Y_centroid']], leaf_size=2)

    ids = exp_tree.query_radius(dataset[['X_centroid', 'Y_centroid']], r=spatial)

    # convert indexes to a list of lists
    edge_index = []
    for i in range(len(ids)):
        for j in range(len(ids[i])):
            edge_index.append([i, ids[i][j]])

    return torch.tensor(edge_index, dtype=torch.long).t()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rm", "--replace_mode", action="store", type=str, choices=["mean", "zero"], default="zero")
    parser.add_argument("-i", "--iterations", action="store", default=10, type=int)
    parser.add_argument("--mode", action="store", type=str, choices=["ip", "exp"], default="ip")
    parser.add_argument("--spatial", "-sp", action="store", type=int, choices=[23, 46, 92, 138, 184], default=46)
    parser.add_argument("-b", "--biopsy", action="store", type=str, required=True, help="The biopsy to run")
    parser.add_argument("-s", "--subsets", action="store", type=int, default=1)

    args = parser.parse_args()
    prepared_data_folder = Path("gnn", "data", args.mode)
    raw_data_folder = Path("data", "tumor_mesmer")
    spatial_radius = args.spatial
    test_biopsy_name = args.biopsy
    mode = args.mode
    subsets: int = args.subsets
    patient = "_".join(test_biopsy_name.split("_")[:2])
    replace_value = args.replace_mode
    iterations = args.iterations

    score_file_name = "scores.csv"

    save_folder, experiment_id = create_results_folder(spatial_radius, mode, replace_value, test_biopsy_name)

    setup_log_file(save_path=save_folder)

    logging.info("Experiment started with the following parameters:")
    logging.info(f"Time:  {str(datetime.datetime.now())}")
    logging.info(f"Patient type: {mode}")
    logging.info(f"Iterations: {iterations}")
    logging.info(f"Replace value: {replace_value}")
    logging.info(f"Subsets: {subsets}")
    logging.info(f"Spatial radius: {spatial_radius}")
    logging.info(f"Test biopsy name: {test_biopsy_name}")
    logging.info(f"Save folder: {save_folder}")
    logging.info(f"Experiment id: {experiment_id}")

    if mode == "ip":
        logging.debug(str(Path(raw_data_folder, test_biopsy_name)))
        raw_test_set: pd.DataFrame = pd.read_csv(str(Path(raw_data_folder, f"{test_biopsy_name}.csv")), header=0)

        logging.debug(
            f"Loading train data using path:  {str(Path(prepared_data_folder, test_biopsy_name, str(spatial_radius), f'train_set.csv'))}")
        raw_test_set: pd.DataFrame = clean_column_names(raw_test_set)
        train_set = pd.read_csv(
            str(Path(prepared_data_folder, test_biopsy_name, str(spatial_radius), f"train_set.csv")),
            header=0)
        train_edge_index = torch.load(
            str(Path(prepared_data_folder, test_biopsy_name, str(spatial_radius), f"train_edge_index.pt")))

        # normalize data
        test_set: pd.DataFrame = raw_test_set[MARKERS]
        test_data: pd.DataFrame = np.log10(test_set + 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_set: pd.DataFrame = pd.DataFrame(scaler.fit_transform(test_set),
                                              columns=test_set.columns)



    else:
        train_set = pd.read_csv(str(Path(prepared_data_folder, f"{patient}_excluded_scaled.csv")),
                                header=0)
        train_edge_index = torch.load(
            str(Path(prepared_data_folder, str(spatial_radius), f"{patient}_excluded_edge_index.pt")))

        raw_test_set = pd.read_csv(str(Path(raw_data_folder, f"{test_biopsy_name}.csv")), header=0)
        raw_test_set = clean_column_names(raw_test_set)

        # normalize data
        test_set: pd.DataFrame = raw_test_set[MARKERS]
        test_set: pd.DataFrame = np.log10(test_set + 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_set: pd.DataFrame = pd.DataFrame(scaler.fit_transform(test_set),
                                              columns=test_set.columns)

    model = GCNAutoencoder(num_features=len(train_set.columns))  # Create new network.
    criterion = torch.nn.L1Loss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
    out = []

    train_node_features = torch.tensor(train_set.values, dtype=torch.float)
    num_train_cells = train_set.shape[0]
    train_mask = np.full(num_train_cells, False)
    train_data: Data = Data(x=train_node_features, train_mask=train_mask, edge_index=train_edge_index,
                            y=train_set['CD45'].values)

    logging.debug("Training model...")
    for epoch in range(100):
        if epoch % 10 == 0:
            logging.debug(f"Epoch: {epoch}")
        loss, h = train(train_data)

    logging.debug("Training complete.")
    # Evaluate model performance on test set.
    model.eval()

    logging.debug("Evaluating...")

    scores = []
    predictions = {}
    # prepare predictions dict
    for i in range(iterations):
        predictions[i] = pd.DataFrame(columns=test_set.columns)

    # Evaluate on subset
    for i in range(1, subsets):
        test_data_sample: pd.DataFrame = test_set.sample(frac=0.7, random_state=random.randint(0, 100000),
                                                         replace=True)

        # select X_centroid and Y_centroid from raw data, by selecting only the cells of the tes_data_sample index
        raw_test_subset = raw_test_set.loc[test_data_sample.index, ['X_centroid', 'Y_centroid']]

        test_sample_node_features = torch.tensor(test_data_sample.values, dtype=torch.float)
        test_sample_mask = np.full(test_data_sample.shape[0], True)

        test_data_sample_edge_index = calculate_edge_indexes(raw_test_subset, spatial_radius)

        train_data: Data = Data(x=train_node_features, train_mask=train_mask, edge_index=train_edge_index,
                                y=train_set['CD45'].values)
        test_data: Data = Data(x=test_sample_node_features, train_mask=test_sample_mask,
                               edge_index=test_data_sample_edge_index,
                               y=test_data_sample['CD45'].values)

        impute_marker(test_data=test_data, subset=i, scores=scores, all_predictions=predictions,
                      store_predictions=False, columns=MARKERS, replace_value=replace_value,
                      iterations=iterations, biopsy_name=test_biopsy_name, save_folder=save_folder,
                      file_name=score_file_name)

    # Evaluate on full test set
    test_node_features = torch.tensor(test_set.values, dtype=torch.float)
    test_mask = np.full(test_set.shape[0], True)
    test_edge_index = calculate_edge_indexes(raw_test_set, spatial_radius)

    test_data: Data = Data(x=test_node_features, train_mask=test_mask, edge_index=test_edge_index,
                           y=test_set['CD45'].values)

    impute_marker(test_data=test_data, subset=0, scores=scores, all_predictions=predictions, store_predictions=True,
                  columns=MARKERS, replace_value=replace_value, iterations=iterations,
                  biopsy_name=test_biopsy_name, save_folder=save_folder, file_name=score_file_name)

    # Convert to df
    scores = pd.DataFrame(scores)

    # Save data
    if Path(f"{save_folder}/{score_file_name}").exists():
        scores = pd.concat([pd.read_csv(f"{save_folder}/{score_file_name}"), scores], axis=0)

    scores.to_csv(f"{save_folder}/{score_file_name}", index=False)
    for key, value in predictions.items():
        value.to_csv(f"{save_folder}/{key}_predictions.csv",
                     index=False)
