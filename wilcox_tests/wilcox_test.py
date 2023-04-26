import os, argparse, json
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bx1", "--bx1", help="First biopsy", required=True)
    parser.add_argument("-bx2", "--bx2", help="Second biopsy", required=True)
    args = parser.parse_args()

    bx1 = pd.read_csv(Path(args.bx1))
    bx2 = pd.read_csv(Path(args.bx2))

    print(bx1)
    print(bx2)
    res = wilcoxon(bx1, bx2)
    print(res.pvalue)
