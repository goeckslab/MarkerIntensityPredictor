import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The test biopsy", required=True)
    parser.add_argument("--mode", choices=["ip", "exp"], required=True, default="ip")

    args = parser.parse_args()
    biopsy_name = args.biopsy
    mode = args.mode

    if mode == "ip":
        load_path = Path("mesmer", "tumor_in_patient_en")
        if biopsy_name[-1] == "1":
            train_biopsy = biopsy_name[:-1] + "2"
        else:
            train_biopsy = biopsy_name[:-1] + "1"

    else:
        load_path = Path("mesmer","tumor_exp_experiment_en")
        train_biopsy = biopsy_name

    for marker in markers:
        predictions = pd.read_csv(str(Path(load_path, train_biopsy, biopsy_name, marker, f"{marker}_predictions.csv")))
        predictions = predictions.rename(columns={"Unnamed: 0": "prediction"})
        print(predictions)
        input()
        ground_truth = pd.read_csv(
            str(Path("data", "tumor_mesmer", "preprocessed", f"{biopsy_name}_preprocessed_dataset.tsv")), sep='\t')

        # extract the quartiles
        quartiles = predictions['prediction'].quantile([0.25, 0.5, 0.75])
