import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import shutil
from ludwig.api import LudwigModel
from statannotations.Annotator import Annotator

save_path = Path("plots", "ludwig", "quartiles")

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']
modes = ["exp", "ip"]


def calculate_quartile_performance(ground_truth: pd.DataFrame, marker: str, predictions: pd.DataFrame, std: int):
    if std > 0:
        # keep only the rows that are within 3 standard deviations of the mean
        ground_truth = ground_truth[
            np.abs(ground_truth[marker] - ground_truth[marker].mean()) <= (std * ground_truth[marker].std())].copy()

    # select all indexes of predictions which are in the ground truth index
    predictions = predictions.loc[ground_truth.index].copy()

    # extract the quartiles
    quartiles = ground_truth.quantile([0.25, 0.5, 0.75])
    # select the rows that are in the quartiles from the predictions and ground truth
    gt_quartile_1 = ground_truth[ground_truth[marker] <= quartiles[marker][0.25]]
    gt_quartile_2 = ground_truth[
        (ground_truth[marker] > quartiles[marker][0.25]) & (
                ground_truth[marker] <= quartiles[marker][0.5])]
    gt_quartile_3 = ground_truth[
        (ground_truth[marker] > quartiles[marker][0.5]) & (
                ground_truth[marker] <= quartiles[marker][0.75])]
    gt_quartile_4 = ground_truth[ground_truth[marker] > quartiles[marker][0.75]]

    pred_quartile_1 = predictions.loc[gt_quartile_1.index]
    pred_quartile_2 = predictions.loc[gt_quartile_2.index]
    pred_quartile_3 = predictions.loc[gt_quartile_3.index]
    pred_quartile_4 = predictions.loc[gt_quartile_4.index]

    # Calculate MAE for all quartiles
    mae_1 = np.mean(np.abs(gt_quartile_1[marker] - pred_quartile_1["prediction"]))
    mae_2 = np.mean(np.abs(gt_quartile_2[marker] - pred_quartile_2["prediction"]))
    mae_3 = np.mean(np.abs(gt_quartile_3[marker] - pred_quartile_3["prediction"]))
    mae_4 = np.mean(np.abs(gt_quartile_4[marker] - pred_quartile_4["prediction"]))
    # mae_all = np.mean(np.abs(ground_truth[marker] - predictions["prediction"]))

    return mae_1, mae_2, mae_3, mae_4, quartiles


def create_outlier_results_file(std: int) -> pd.DataFrame:
    results = []
    for mode in modes:
        for train_biopsy_name in biopsies:
            if mode == "ip":
                if spatial == 0:
                    load_path = Path("mesmer", "tumor_in_patient")
                else:
                    load_path = Path("mesmer", f"tumor_in_patient_sp_{spatial}")
                if train_biopsy_name[-1] == "1":
                    test_biopsy_name = train_biopsy_name[:-1] + "2"
                else:
                    test_biopsy_name = train_biopsy_name[:-1] + "1"

            else:
                if spatial == 0:
                    load_path = Path("mesmer", "tumor_exp_patient")
                else:
                    load_path = Path("mesmer", f"tumor_exp_patient_sp_{spatial}")
                test_biopsy_name = train_biopsy_name

            # load ground truth
            if spatial == 0:
                ground_truth = pd.read_csv(
                    str(Path("data", "tumor_mesmer", "preprocessed",
                             f"{test_biopsy_name}_preprocessed_dataset.tsv")), sep='\t')
            else:
                ground_truth = pd.read_csv(
                    str(Path("data", f"tumor_mesmer_sp_{spatial}", "preprocessed",
                             f"{test_biopsy_name}_preprocessed_dataset.tsv")), sep='\t')

            for marker in markers:
                experiment_folder = Path(load_path, train_biopsy_name, marker, "results")
                for marker_results_dir, _, _ in os.walk(experiment_folder):
                    for experiment_run, _, _ in os.walk(marker_results_dir):
                        if "experiment_run" not in experiment_run:
                            continue

                        if not Path(experiment_run, "predictions.csv").exists():
                            model = None
                            try:
                                model = LudwigModel.load(str(Path(experiment_run, 'model')))
                            except:
                                continue

                            print(
                                f"Predictions file does not exist. Creating predictions for {str(Path(marker_results_dir, experiment_run))}")
                            # drop current marker from ground truth
                            predictions = pd.DataFrame(model.predict(ground_truth.drop(marker, axis=1))[0])
                            # rename the prediction column
                            predictions = predictions.rename(columns={f"{marker}_predictions": "prediction"})
                            predictions.to_csv(str(Path(experiment_run, "predictions.csv")), index=False)

                        else:
                            print("Loading predictions...")
                            predictions = pd.read_csv(str(Path(experiment_run, "predictions.csv")))

                        mae_1, mae_2, mae_3, mae_4, quartiles = calculate_quartile_performance(
                            ground_truth=ground_truth,
                            marker=marker,
                            predictions=predictions,
                            std=std)
                        # add these values to a dataframe
                        results.append(pd.DataFrame(
                            {"MAE": [mae_1, mae_2, mae_3, mae_4], "Quartile": ["Q1", "Q2", "Q3", "Q4"],
                             "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                           quartiles[marker][0.75]], "Marker": marker,
                             "Biopsy": test_biopsy_name,
                             "Mode": mode,
                             "Load Path": str(Path(marker_results_dir, experiment_run)),
                             "Std": std
                             }))

    results = pd.concat(results)
    if std > 0:
        results.to_csv(str(Path(save_path, f"{std}_std_accuracy.csv")), index=False)
    else:
        results.to_csv(str(Path(save_path, f"accuracy.csv")), index=False)

    return results


def create_boxen_plot_ip_vs_exp(results: pd.DataFrame, metric: str, title: str):
    # plot the quartiles
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, hue_order=["IP", "EXP"],
                       palette={"IP": "lightblue", "EXP": "orange"})
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Light GBM \n{metric.upper()} per quartile\nAll Biopsies", fontsize=20, y=1.3)

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        # (("Q1", "IP"), ("Q1", "EXP")),
        # (("Q2", "IP"), ("Q2", "EXP")),
        # (("Q3", "IP"), ("Q3", "EXP")),
        # (("Q4", "IP"), ("Q4", "EXP")),
        # (("All", "IP"), ("All", "EXP")),
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        # (("Q4", "IP"), ("All", "IP")),
        (("Q1", "EXP"), ("Q2", "EXP")),
        (("Q2", "EXP"), ("Q3", "EXP")),
        (("Q3", "EXP"), ("Q4", "EXP")),
        # (("Q4", "EXP"), ("All", "EXP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    plt.savefig(str(Path(save_path, f"{title}_accuracy.png")))


def create_boxen_plot_std(results: pd.DataFrame, metric: str, title: str, mode: str):
    # plot the quartiles
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Std", data=results,
                       hue_order=[0, 2, 3])
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())
    # ax.set_title(f"Light GBM \n{metric.upper()} per quartile\nAll Biopsies\nMode: {mode.upper()}", fontsize=20, y=1.3)

    hue = "Std"
    hue_order = [0, 2, 3]
    pairs = [
        (("Q1", 2), ("Q1", 3)),
        (("Q2", 2), ("Q2", 3)),
        (("Q3", 2), ("Q3", 3)),
        (("Q4", 2), ("Q4", 3)),
        # (("All", 2), ("All", 3)),
        (("Q1", 0), ("Q1", 2)),
        (("Q2", 0), ("Q2", 2)),
        (("Q3", 0), ("Q3", 2)),
        (("Q4", 0), ("Q4", 2)),
        # (("All", 0), ("All", 2)),
        (("Q1", 0), ("Q1", 3)),
        (("Q2", 0), ("Q2", 3)),
        (("Q3", 0), ("Q3", 3)),
        (("Q4", 0), ("Q4", 3)),
        # (("All", 0), ("All", 3)),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    # increase font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.box(False)
    # remove x and y label
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(str(Path(save_path, f"{title}_accuracy.png")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", "--spatial", choices=[0, 23, 46, 92, 138, 184], default=0, type=int)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="MAE", type=str)

    args = parser.parse_args()
    spatial: int = args.spatial
    metric: str = args.metric
    # mode = args.mode

    save_path = Path(save_path, str(spatial))

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    print("Creating outlier results...")
    outlier_results: pd.DataFrame = create_outlier_results_file(std=0)
    # change mode to uppercase
    outlier_results["Mode"] = outlier_results["Mode"].str.upper()

    print("Creating 3 std outlier results...")
    std_3_outlier_results: pd.DataFrame = create_outlier_results_file(std=3)
    # change mode to uppercase
    std_3_outlier_results["Mode"] = std_3_outlier_results["Mode"].str.upper()

    print("Creating 2 std outlier results...")
    std_2_outlier_results: pd.DataFrame = create_outlier_results_file(std=2)
    # change mode to uppercase
    std_2_outlier_results["Mode"] = std_3_outlier_results["Mode"].str.upper()

    create_boxen_plot_ip_vs_exp(results=std_3_outlier_results, metric=metric, title=f"3_std_outlier_{metric}")
    create_boxen_plot_ip_vs_exp(results=std_2_outlier_results, metric=metric, title=f"2_std_outlier_{metric}")
    create_boxen_plot_ip_vs_exp(results=outlier_results, metric=metric, title=f"outlier_{metric}")

    df = pd.concat([outlier_results, std_2_outlier_results, std_3_outlier_results])

    for mode in df["Mode"].unique():
        temp_df = df[df["Mode"] == mode]
        create_boxen_plot_std(results=temp_df, metric=metric, title=f"{mode}_{metric}_std", mode=mode)
