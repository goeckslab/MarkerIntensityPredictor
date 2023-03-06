import json

import pandas as pd
import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--biopsy", help="The biopsy", required=True)
    args = parser.parse_args()
    scores = []
    for marker in markers:
        path = Path(args.biopsy, f"{marker}", "evaluate", args.biopsy, "test_statistics.json")
        f = open(path)
        data = json.load(f)
        scores.append(
            {
                "Marker": marker,
                "MAE": data[marker]['mean_absolute_error'],
                "MSE": data[marker]['mean_squared_error'],
                "RMSE": data[marker]['root_mean_squared_error'],
                "Biopsy": args.biopsy,
                "Panel": "Tumor",
                "Type": "OP",
                "Segmentation": "Unmicst + S3",
                "SNR": 0,
            }
        )

    scores = pd.DataFrame.from_records(scores)

    scores.to_csv(Path(args.biopsy, f"{args.biopsy}_scores.csv"), index=False)
