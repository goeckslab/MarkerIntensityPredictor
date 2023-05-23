import os
import numpy as np
from sklearn.linear_model import ElasticNetCV
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

base_path = Path("en")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The biopsy which is evaluated", action="store", required=True)
    parser.add_argument("--mode", help="The mode", choices=["ip", "exp"], required=True, default="ip")

    args = parser.parse_args()
    test_biopsy = args.biopsy
    mode = args.mode
    patient = "_".join(test_biopsy.split("_")[:2])
    print(patient)

    experiment_id = 0
    base_path = Path(base_path, mode, test_biopsy, "experiment_run")
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

    if mode == "ip":
        if test_biopsy[-1] == "2":
            train_biopsy = test_biopsy[:-1] + "1"
        else:
            train_biopsy = test_biopsy[:-1] + "2"

        test_df = pd.read_csv(Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy}_preprocessed_dataset.tsv"),
                              sep="\t", header=0)
        train_df = pd.read_csv(Path("data", "tumor_mesmer", "preprocessed", f"{train_biopsy}_preprocessed_dataset.tsv"),
                               sep="\t", header=0)

    elif mode == "exp":
        train_biopsy = "EXP"
        test_df = pd.read_csv(Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy}_preprocessed_dataset.tsv"),
                              sep="\t", header=0)
        train_df = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t", header=0)
    else:
        raise ValueError("Mode not recognized")

    results_per_marker = []
    importance_per_marker = []
    for test_marker in test_df.columns:
        X = pd.DataFrame()
        X_test = pd.DataFrame()
        for train_marker in train_df.columns:
            if train_marker == test_marker:
                continue
            X[train_marker] = train_df[train_marker]
            y = train_df[test_marker]

            elastic_net = ElasticNetCV(cv=5, random_state=0)
            elastic_net.fit(X, y)

            X_test[train_marker] = test_df[train_marker]
            y_test = test_df[test_marker]

            y_hat = elastic_net.predict(X_test)
            y_hat_df = pd.DataFrame(y_hat, columns=[test_marker])

            if len(elastic_net.coef_) == 15:
                importance = pd.DataFrame(
                    {'Markers': list(X.columns),
                     'Importance': list(elastic_net.coef_),
                     })
                importance["Target"] = test_marker
                importance["Patient"] = patient
                importance["Biopsy"] = test_biopsy
                importance["Experiment"] = experiment_id
                importance["Type"] = mode
                importance["Model"] = "EN"
                importance_per_marker.append(importance)

            # y_hat_df.to_csv(Path(save_path, f"{args.marker}_predictions.csv"), index=False, header=False)

            results_per_marker.append({
                "Patient": patient,
                "MAE": mean_absolute_error(y_test, y_hat),
                "RMSE": mean_squared_error(y_test, y_hat, squared=False),
                "Marker": train_marker,
                "Target": test_marker,
                "Model": "EN",
                "Experiment": experiment_id,
                "Type": mode,
                "Biopsy": test_biopsy,
                "Full Panel": 1 if len(elastic_net.coef_) == 15 else 0
            })

    results_per_marker = pd.DataFrame(results_per_marker)
    results_per_marker.to_csv(Path(save_path, f"results.csv"), index=False, header=True)

    importance_per_marker = pd.concat(importance_per_marker)
    importance_per_marker.to_csv(Path(save_path, f"importance_per_marker.csv"), index=False, header=True)

    # heatmap
    fig = plt.figure(dpi=200, figsize=(10, 10))
    ax = sns.heatmap(train_df.corr(), annot=True)
    ax.set_title(f"Train {train_biopsy}")
    fig.savefig(Path(save_path, f"train_heatmap.png"))

    fig = plt.figure(dpi=200, figsize=(10, 10))
    ax = sns.heatmap(test_df.corr(), annot=True)
    ax.set_title(f"Test {test_biopsy}")
    fig.savefig(Path(save_path, f"test_heatmap.png"))

    # importance

    # select only

    fig = plt.figure(dpi=200, figsize=(10, 10))
    ax = sns.scatterplot(x="Markers", y="Importance", data=importance_per_marker, hue="Target")
    ax.set_title(f"Importance {test_biopsy}")
    fig.savefig(Path(save_path, f"importance.png"))

    results = []
    # for each marker select the 3 most predicitve markers
    for target in importance_per_marker["Target"].unique():
        most_important_markers = importance_per_marker[importance_per_marker["Target"] == target].sort_values(
            "Importance", ascending=False)[:3]

        X = train_df[most_important_markers["Markers"].values]
        y = train_df[target]

        elastic_net = ElasticNetCV(cv=5, random_state=0)
        elastic_net.fit(X, y)

        X_test = test_df[most_important_markers["Markers"].values]
        y_test = test_df[target]

        y_hat = elastic_net.predict(X_test)
        y_hat_df = pd.DataFrame(y_hat, columns=[target])

        # select mae for target from results_per_marker where train marker is ER
        target_mae_full_panel = \
            results_per_marker[(results_per_marker["Target"] == target) & (results_per_marker["Full Panel"] == 1)][
                "MAE"].values[0]

        target_rmse_full_panel = \
            results_per_marker[(results_per_marker["Target"] == target) & (results_per_marker["Full Panel"] == 1)][
                "RMSE"].values[0]

        results.append({
            "Patient": patient,
            "MAE": mean_absolute_error(y_test, y_hat),
            "RMSE": mean_squared_error(y_test, y_hat, squared=False),
            "Target": target,
            "MOI": most_important_markers["Markers"].values,
            "MAE_FP": target_mae_full_panel,
            "RMSE_FP": target_rmse_full_panel,
            "Biopsy": test_biopsy,
            "Mode": mode,
        })

    results = pd.DataFrame(results)
    results.to_csv(Path(save_path, f"results_top_3.csv"), index=False, header=True)
