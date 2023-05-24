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


def create_results_file() -> pd.DataFrame:
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

                        pred_quartile_1 = predictions.iloc[gt_quartile_1.index]
                        pred_quartile_2 = predictions.iloc[gt_quartile_2.index]
                        pred_quartile_3 = predictions.iloc[gt_quartile_3.index]
                        pred_quartile_4 = predictions.iloc[gt_quartile_4.index]

                        # Calculate MAE for all quartiles
                        mae_1 = np.mean(np.abs(gt_quartile_1[marker] - pred_quartile_1["prediction"]))
                        mae_2 = np.mean(np.abs(gt_quartile_2[marker] - pred_quartile_2["prediction"]))
                        mae_3 = np.mean(np.abs(gt_quartile_3[marker] - pred_quartile_3["prediction"]))
                        mae_4 = np.mean(np.abs(gt_quartile_4[marker] - pred_quartile_4["prediction"]))
                        mae_all = np.mean(np.abs(ground_truth[marker] - predictions["prediction"]))

                        # add these values to a dataframe
                        results.append(pd.DataFrame(
                            {"MAE": [mae_1, mae_2, mae_3, mae_4, mae_all], "Quartile": ["Q1", "Q2", "Q3", "Q4", "All"],
                             "Threshold": [quartiles[marker][0.25], quartiles[marker][0.5], quartiles[marker][0.75],
                                           quartiles[marker][0.75], "All"], "Marker": marker,
                             "Biopsy": test_biopsy_name,
                             "Mode": mode}))

    results = pd.concat(results)
    results.to_csv(str(Path(save_path, f"accuracy.csv")), index=False)

    return results


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

    print("Creating results file...")
    results: pd.DataFrame = create_results_file()
    # change mode to uppercase
    results["Mode"] = results["Mode"].str.upper()

    # plot the quartiles
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, palette="Set2")
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Light GBM \n{metric.upper()} per quartile\nAll Biopsies", fontsize=20, y=1.3)

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        (("Q1", "IP"), ("Q1", "EXP")),
        (("Q2", "IP"), ("Q2", "EXP")),
        (("Q3", "IP"), ("Q3", "EXP")),
        (("Q4", "IP"), ("Q4", "EXP")),
        (("All", "IP"), ("All", "EXP")),
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        (("Q4", "IP"), ("All", "IP")),
        (("Q1", "EXP"), ("Q2", "EXP")),
        (("Q2", "EXP"), ("Q3", "EXP")),
        (("Q3", "EXP"), ("Q4", "EXP")),
        (("Q4", "EXP"), ("All", "EXP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4", "All"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    plt.savefig(str(Path(save_path, f"accuracy.png")))
