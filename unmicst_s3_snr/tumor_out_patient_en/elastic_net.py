import os
import numpy as np
from sklearn.linear_model import ElasticNetCV
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train data", required=True)
    parser.add_argument("--test", help="test data", required=True)
    parser.add_argument("--marker", help="marker", required=True)

    args = parser.parse_args()

    train_df = pd.read_csv(args.train, sep="\t", header=0)
    test_df = pd.read_csv(args.test, sep="\t", header=0)

    X = train_df.drop(columns=[args.marker])
    y = train_df[args.marker]

    elastic_net = ElasticNetCV(cv=5, random_state=0)
    elastic_net.fit(X, y)

    X_test = test_df.drop(columns=[args.marker])
    y_test = test_df[args.marker]

    y_hat = elastic_net.predict(X_test)
    y_hat_df = pd.DataFrame(y_hat, columns=[args.marker])

    y_hat_df.to_csv(f"{args.marker}_y_hat.csv", index=False)

    data = {
        "patient": " ".join(Path(args.test).stem.split("_")[0:3]),
        "mse": mean_squared_error(y_test, y_hat),
        "mae": mean_absolute_error(y_test, y_hat),
        "mape": mean_absolute_percentage_error(y_test, y_hat),
        "marker": args.marker,
        "model": "elastic_net",
        "rmse": mean_squared_error(y_test, y_hat, squared=False),
        # "rmspe": np.sqrt(np.mean(np.square(((y_test - y_hat_df[args.marker]) / y_test)), axis=0))
    }

    with open('evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
