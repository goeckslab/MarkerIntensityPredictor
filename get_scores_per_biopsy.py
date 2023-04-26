import json

import pandas as pd
import argparse
from pathlib import Path

markers = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK',
           'EGFR', 'ER']

scores_directory = Path("data/scores")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="The biopsy", required=True)
    args = parser.parse_args()

    train_biopsy = args.biopsy
    train_biopsy_name = Path(args.biopsy).stem
    data_path = Path(args.biopsy).parent

    in_or_out = "in_patient" if "in_patient" in str(data_path) else "out_patient"

    if in_or_out == "in_patient":
        # replace last number with 1 if last number is 2
        if train_biopsy_name[-1] == "2":
            test_biopsy = train_biopsy_name[:-1] + "1"
        else:
            test_biopsy = train_biopsy_name[:-1] + "2"

    else:
        test_biopsy = Path(args.biopsy).stem

    type = "IP" if in_or_out == "in_patient" else "OP"
    segmentation = "Unmicst + S3" if "unmicst" in str(data_path) else "Mesmer"
    snr = 0 if "non_snr" in str(data_path) else 1

    if "_en_6_2" in str(data_path):
        mode = "EN_6_2"
    elif "_en" in str(data_path):
        mode = "EN"
    else:
        mode = "Ludwig"
    hyper = 1 if "_hyper" in str(data_path) else 0

    if "_sp" in str(data_path):
        splits = str(data_path).split("_")
        fe = f"{splits[-2]}_{splits[-1]}"
    else:
        fe = "None"

    scores_directory = Path(scores_directory, segmentation)

    scores_directory = Path(scores_directory, "in_patient") if type == "IP" else Path(scores_directory,
                                                                                      "out_patient")
    if segmentation != "Mesmer":
        scores_directory = Path(scores_directory, "snr") if snr else Path(scores_directory, "non_snr")

    if not hyper and fe == "None":
        scores_directory = Path(scores_directory, mode)
    elif not hyper and fe != "None":
        scores_directory = Path(scores_directory, f"{mode}_{fe}")
    else:
        scores_directory = Path(scores_directory, f"{mode}_Hyper")

    print(scores_directory)
    if not scores_directory.exists():
        scores_directory.mkdir(parents=True, exist_ok=True)

    scores = []
    for marker in markers:
        path = Path(args.biopsy, f"{marker}", "evaluate", test_biopsy, "test_statistics.json")
        if not path.exists():
            path = Path(args.biopsy, test_biopsy, marker, "evaluation.json")
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")

        f = open(path)
        data = json.load(f)

        if mode == "EN" and test_biopsy == "9_2_2" and type == "IP" and marker == "AR":
            assert data['mean_absolute_error'] == 0.04704521207831636, "AR MAE should be 0.04704521207831636"

        scores.append(
            {
                "Marker": marker,
                "MAE": data[marker]['mean_absolute_error'] if "_en" not in str(data_path) else data[
                    'mean_absolute_error'],
                "MSE": data[marker]['mean_squared_error'] if "_en" not in str(data_path) else data[
                    'mean_squared_error'],
                "RMSE": data[marker]['root_mean_squared_error'] if "_en" not in str(data_path) else data[
                    'root_mean_squared_error'],
                "Biopsy": test_biopsy,
                "Panel": "Tumor",
                "Type": type,
                "Segmentation": segmentation,
                "SNR": snr,
                "FE": fe,
                "Mode": mode,
                "Hyper": hyper
            }
        )

    scores = pd.DataFrame.from_records(scores)

    scores.to_csv(Path(scores_directory,
                       f"{test_biopsy}_{type}_{'_'.join(segmentation.split(' '))}_{snr}_{fe}_{mode}_{hyper}_scores.csv"),
                  index=False)
