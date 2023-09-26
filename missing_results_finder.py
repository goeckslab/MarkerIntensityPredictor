import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="biopsy to check results", required=True)
    args = parser.parse_args()

    biopsy = args.biopsy

    missing_markers = []

    for marker in markers:
        path = Path(args.biopsy, f"{marker}", "evaluate", Path(biopsy).stem, "test_statistics.json")
        if not path.exists():
            missing_markers.append(marker)

    if len(missing_markers) == 0:
        print("All markers are present")
    else:
        print(missing_markers)
