import argparse
import pandas as pd
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

results_path = Path("data/absolute_error_per_cell/ae/")

if __name__ == "__main__":

    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ground_truth", help="the ground truth values")
    parser.add_argument("-p", "--predictions", help="the predicted values")
    args = parser.parse_args()
    predictions = args.predictions
    ground_truth = args.ground_truth

    predicted_biopsy = predictions.split("/")[-2]
    patient_type = predictions.split("/")[-3]
    results_path = Path(results_path, patient_type)

    predicted = pd.read_csv(predictions, delimiter=',', header=0)
    source = pd.read_csv(ground_truth, delimiter=',', header=0)

    results = pd.DataFrame()
    for marker in markers:
        temp_df = pd.DataFrame()
        temp_df["source"] = source[marker].values
        temp_df["predicted"] = predicted[marker].values

        # calculate absolute error per cell
        results[marker] = abs(temp_df["source"] - temp_df["predicted"])

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    results.to_csv(f"{Path(results_path)}/{predicted_biopsy}.csv", index=False)
