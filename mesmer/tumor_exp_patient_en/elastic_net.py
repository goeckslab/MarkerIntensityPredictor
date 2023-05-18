import os, shutil
import numpy as np
from sklearn.linear_model import ElasticNetCV
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

base_path = Path("experiment_run")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train data", required=True)
    parser.add_argument("--test", help="test data", required=True)
    parser.add_argument("--marker", help="marker", required=True)

    args = parser.parse_args()

    experiment_id = 0
    # save_path = Path(str(base_path) + "_" + str(experiment_id))
    save_path = Path(str(base_path))
    while Path(save_path).exists():
        # save_path = Path(str(base_path) + "_" + str(experiment_id))
        # experiment_id += 1
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

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

    y_hat_df.to_csv(Path(save_path, f"{args.marker}_predictions.csv"), index=False, header=False)

    data = {
        "patient": " ".join(Path(args.test).stem.split("_")[0:3]),
        "mean_squared_error": mean_squared_error(y_test, y_hat),
        "mean_absolute_error": mean_absolute_error(y_test, y_hat),
        "root_mean_squared_error": mean_squared_error(y_test, y_hat, squared=False),
        "mape": mean_absolute_percentage_error(y_test, y_hat),
        "marker": args.marker,
        "model": "elastic_net",
        # "rmspe": np.sqrt(np.mean(np.square(((y_test - y_hat_df[args.marker]) / y_test)), axis=0))
    }

    with open(Path(save_path, 'evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
