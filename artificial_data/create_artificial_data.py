import numpy as np
import pandas as pd
import argparse
from ludwig.api import LudwigModel
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def gen_random(n: int, d: int, covar: float) -> np.ndarray:
    """
    Paramters
    ---------
    n : int
        number of samples generated
    d : int
        dimensionality of samples
    covar : float
        uniform covariance for samples

    Returns
    -------
    samples : np.ndarray
        samples in as (n, d)-matrix
    """
    cov_mat = np.ones((d, d)) * covar
    np.fill_diagonal(cov_mat, 1)
    offset = np.zeros(d)

    return np.random.multivariate_normal(offset, cov_mat, size=n)


def gen_totally_random(n: int, columns):
    # Dictionary to hold data for each column
    data = {}

    # Generate random data for each column
    for protein in columns:
        data[protein] = np.random.uniform(-1, 1, n)

    # Create DataFrame
    return pd.DataFrame(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["EN", "LGBM"])
    args = parser.parse_args()

    selected_model: str = args.model

    correlations_to_test = [None, 0, 0.5, 0.9]
    errors = []
    all_predictions = []
    for i in tqdm(range(30)):
        for correlation in correlations_to_test:
            if correlation is not None:
                # generate random data with same amount of cells like the ground truth
                v = gen_random(n=3000, d=len(SHARED_MARKERS), covar=correlation)
                df = pd.DataFrame(v, columns=SHARED_MARKERS)
            else:
                # Number of data points
                # Generate random numbers between -1 and 1
                df = gen_totally_random(3000, SHARED_MARKERS)

                # scale data using minmax scaler
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=SHARED_MARKERS)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            if selected_model == "EN":
                for marker in SHARED_MARKERS:
                    # split train and test_df
                    X_train = train_df.drop(marker, axis=1)
                    y_train = train_df[marker]

                    X_test = test_df.drop(marker, axis=1)
                    y_test = test_df[marker]

                    # fit elastic net model
                    model = ElasticNetCV(cv=5, random_state=0)
                    model.fit(X_train, y_train)

                    # predict on test data
                    predictions = model.predict(X_test)

                    # save predictions
                    errors.append({
                        "Marker": marker,
                        "MAE": mean_absolute_error(y_test, predictions),
                        "RMSE": mean_squared_error(y_test, predictions),
                        "Correlation": -1 if correlation is None else correlation,
                        "Experiment": i
                    })

            else:
                for marker in SHARED_MARKERS:
                    inputs = [{"name": c, "type": "numerical"} for c in SHARED_MARKERS if c != marker]
                    outputs = [{"name": marker, "type": "numerical"}]
                    config = {
                        "input_features": inputs,
                        "output_features": outputs
                    }

                    model = LudwigModel(config)

                    train_stats, _, model_dir = model.train(train_df)

                    # predict on data without the marker we want to predict

                    predictions = model.predict(dataset=test_df)
                    predictions = predictions[0]
                    # rename _predictions column with marker name
                    predictions = predictions.rename(columns={f"{marker}_predictions": marker})

                    # save predictions
                    errors.append({
                        "Marker": marker,
                        "MAE": mean_absolute_error(test_df[marker], predictions[marker]),
                        "RMSE": mean_squared_error(test_df[marker], predictions[marker], squared=False),
                        "Correlation": -1 if correlation is None else correlation,
                        "Experiment": i
                    })

        # save predictions
    errors = pd.DataFrame(errors)
    if selected_model == "EN":
        errors.to_csv(Path("en_errors.csv"), index=False)
    else:
        errors.to_csv(Path("lgbm_errors.csv"), index=False)
