import json

import pandas as pd
import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tb", "--train_biopsy", help="The biopsy")
    args = parser.parse_args()

    train_biopsy = args.train_biopsy
    # replace last number with 1 if last number is 2
    if train_biopsy[-1] == "2":
        test_biopsy = train_biopsy[:-1] + "1"
    else:
        test_biopsy = train_biopsy[:-1] + "2"

    mae_scores = []
    for marker in markers:
        path = Path(train_biopsy, f"{marker}", "evaluate", test_biopsy, "test_statistics.json")
        f = open(path)
        data = json.load(f)
        mae_scores.append(
            {
                "Marker": marker,
                "Score": data[marker]['mean_absolute_error'],
                "Biopsy": test_biopsy,
                "Panel": "Tumor",
                "Type": "IP",
                "Segmentation": "Unmicst + S3",
            }
        )

    mae_scores = pd.DataFrame.from_records(mae_scores)

    mae_scores.to_csv(Path(train_biopsy, f"{test_biopsy}_mae_scores.csv"), index=False)
