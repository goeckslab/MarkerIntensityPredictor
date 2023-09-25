import json

import pandas as pd
import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The biopsy")
    args = parser.parse_args()

    train_biopsy = args.biopsy
    # replace last number with 1 if last number is 2
    if train_biopsy[-1] == "2":
        test_biopsy = train_biopsy[:-1] + "1"
    else:
        test_biopsy = train_biopsy[:-1] + "2"

    scores = []
    for marker in markers:
        path = Path(train_biopsy, f"{marker}", "evaluation.json")
        f = open(path)
        data = json.load(f)
        scores.append(
            {
                "Marker": marker,
                "MAE": data['mean_absolute_error'],
                "MSE": data['mean_squared_error'],
                "RMSE": data['root_mean_squared_error'],
                "Biopsy": test_biopsy,
                "Panel": "Tumor",
                "Type": "IP",
                "Segmentation": "Unmicst + S3",
                "SNR": 1
            }
        )

    scores = pd.DataFrame.from_records(scores)

    scores.to_csv(Path(train_biopsy, f"{test_biopsy}_scores.csv"), index=False)
