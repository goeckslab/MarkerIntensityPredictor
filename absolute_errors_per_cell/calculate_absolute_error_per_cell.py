import argparse
import pandas as pd
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

results_path = Path("data/absolute_error_per_cell")

if __name__ == "__main__":

    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", help="the folder where to find the prediction")
    args = parser.parse_args()
    predictions = args.predictions

    ip_patient = True if "in_patient" in str(predictions) else False
    source_biopsy = Path(predictions).stem
    results_path = Path(results_path, str([x for x in str(predictions).strip().split("/") if x][-2]))

    if ip_patient and "_en" not in predictions:
        # replace last number with 1 if last number is 2
        if source_biopsy[-1] == "2":
            predicted_biopsy = source_biopsy[:-1] + "1"
        else:
            predicted_biopsy = source_biopsy[:-1] + "2"
            assert predicted_biopsy[-1] == "2"
    else:
        predicted_biopsy = source_biopsy
        assert predicted_biopsy[-1] == source_biopsy[
            -1], "predicted biopsy and source biopsy should have the same last number"

    source_path = Path(f"data/tumor_mesmer/preprocessed/{predicted_biopsy}_preprocessed_dataset.tsv")
    # load source and predicted marker expressions
    source = pd.read_csv(source_path, sep="\t")
    results = pd.DataFrame()
    for marker in markers:
        # load predicted marker expression
        load_path = Path(args.predictions, marker, "evaluate", predicted_biopsy, f"{marker}_predictions.csv")
        if load_path.exists():
            predicted = pd.read_csv(load_path, header=None)
        else:
            predicted = pd.read_csv(Path(args.predictions, marker, f"{marker}_y_hat.csv"))

        temp_df = pd.DataFrame()
        temp_df["source"] = source[marker].values
        temp_df["predicted"] = predicted.values

        # calculate absolute error per cell
        results[marker] = abs(temp_df["source"] - temp_df["predicted"])

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    results.to_csv(f"{Path(results_path)}/{predicted_biopsy}.csv", index=False)
