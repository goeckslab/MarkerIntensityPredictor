import os
import random
import shutil

import numpy as np
from sklearn.linear_model import ElasticNetCV
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

base_path = Path("experiment_run")
l1_ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train data", required=True)
    parser.add_argument("--test", help="test data", required=True)
    parser.add_argument("--marker", help="marker", required=True)

    args = parser.parse_args()

    experiment_id = 0
    save_path = Path(str(base_path) + "_" + str(experiment_id))
    #save_path = Path(str(base_path))
    while Path(save_path).exists():
        experiment_id += 1
        save_path = Path(str(base_path) + "_" + str(experiment_id))


    save_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train, sep="\t", header=0)
    test_df = pd.read_csv(args.test, sep="\t", header=0)

    X = train_df.drop(columns=[args.marker])
    y = train_df[args.marker]

    # generate random state
    l1_ratio = random.choice(l1_ratios)
    elastic_net = ElasticNetCV(cv=5, random_state=random.randint(0, 10000), l1_ratio=l1_ratio)
    elastic_net.fit(X, y)

    X_test = test_df.drop(columns=[args.marker])
    y_test = test_df[args.marker]

    y_hat = elastic_net.predict(X_test)
    y_hat_df = pd.DataFrame(y_hat, columns=[args.marker])

    y_hat_df.to_csv(Path(save_path, f"{args.marker}_predictions.csv"), index=False, header=False)

    data = {
        "biopsy": " ".join(Path(args.test).stem.split("_")[0:3]),
        "mode": "IP",
        "mean_squared_error": mean_squared_error(y_test, y_hat),
        "mean_absolute_error": mean_absolute_error(y_test, y_hat),
        "root_mean_squared_error": mean_squared_error(y_test, y_hat, squared=False),
        "mape": mean_absolute_percentage_error(y_test, y_hat),
        "marker": args.marker,
        "model": "elastic_net",
        "l1_ratio": l1_ratio,
        "experiment_id": experiment_id
        # "rmspe": np.sqrt(np.mean(np.square(((y_test - y_hat_df[args.marker]) / y_test)), axis=0))
    }

    with open(Path(save_path, "evaluation.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
