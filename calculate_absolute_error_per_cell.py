import argparse
import pandas as pd
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

results_path = Path("data/absolute_error_per_cell")

if __name__ == "__main__":

    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", help="where to find the prediction")
    args = parser.parse_args()
    predictions = args.predictions

    ip_patient = True if "in_patient" in str(predictions) else False
    source_biopsy = Path(predictions).stem
    results_path = Path(results_path, str(predictions).split("/")[-2])

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    if ip_patient:
        # replace last number with 1 if last number is 2
        if source_biopsy[-1] == "2":
            predicted_biopsy = source_biopsy[:-1] + "1"
        else:
            predicted_biopsy = source_biopsy[:-1] + "2"
    else:
        predicted_biopsy = source_biopsy

    # load source and predicted marker expressions
    source = pd.read_csv(Path(f"data/tumor_mesmer/preprocessed/{predicted_biopsy}_preprocessed_dataset.tsv"), sep="\t")

    results = pd.DataFrame()
    for marker in markers:
        # load predicted marker expression
        predicted = pd.read_csv(
            Path(args.predictions, marker, "evaluate", predicted_biopsy, f"{marker}_predictions.csv"), header=None)
        temp_df = pd.DataFrame()
        temp_df["source"] = source[marker]
        temp_df["predicted"] = predicted
        # calculate absolute error per cell
        results[marker] = abs(temp_df["source"] - temp_df["predicted"])

    results.to_csv(str(f"{Path(results_path)}/{predicted_biopsy}.csv"), index=False)
