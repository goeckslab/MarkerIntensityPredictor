import json

import pandas as pd
import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--biopsy", help="The biopsy")
    parser.add_argument("--alt_biopsy", help="The biopsy")
    args = parser.parse_args()
    mae_scores = []
    for marker in markers:
        path = Path(args.biopsy, f"{marker}", "evaluate", args.alt_biopsy, "test_statistics.json")
        f = open(path)
        data = json.load(f)
        mae_scores.append(
            {
                "Marker": marker,
                "Score": data[marker]['mean_absolute_error']
            }
        )

    mae_scores = pd.DataFrame.from_records(mae_scores)

    mae_scores.to_csv(Path(args.biopsy, f"{args.biopsy}_mae_scores.csv"), index=False)
