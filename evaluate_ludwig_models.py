import os, argparse
from pathlib import Path
from ludwig.api import LudwigModel
import pandas as pd


def create_scores_dir(biopsy_path: str, combination: str):

    if "_sp" in str(biopsy_path):
        splits = str(biopsy_path).split("_")
        fe = f"{splits[-2]}_{splits[-1]}"
    else:
        fe = "None"

    scores_directory = Path("scores/Mesmer")

    scores_directory = Path(scores_directory, combination)

    if fe == "None":
        scores_directory = Path(scores_directory, combination)
    else:
        scores_directory = Path(scores_directory, f"{combination}_{fe}")

    print(scores_directory)
    input()
    # if not scores_directory.exists():
    #    scores_directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", type=str, required=True, help="path to biopsy")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="path to dataset")
    args = parser.parse_args()

    biopsy_path: str = args.biopsy
    dataset: str = args.dataset
    train_biopsy_name = Path(dataset).stem

    scores = []

    if 'in_patient' in str(biopsy_path):
        combination = 'IP'
    elif 'out_patient' in str(biopsy_path):
        combination = 'OP'
    elif '_6_2' in str(biopsy_path) or '_exp' in str(biopsy_path):
        combination = 'EXP'
    else:
        raise ValueError("Unknown mode")

    fe = 1 if '_sp' in str(biopsy_path) else 0

    if combination == "IP":
        # replace last number with 1 if last number is 2
        if biopsy_path[-1] == "2":
            test_biopsy = biopsy_path[:-1] + "1"
        else:
            test_biopsy = Path(biopsy_path[:-1] + "2").stem

    else:
        test_biopsy = Path(biopsy_path).stem

    print(train_biopsy_name)
    print(test_biopsy)
    create_scores_dir(biopsy_path=biopsy_path, combination=combination)

    # iterate through all the models
    for root, dirs, files in os.walk(biopsy_path):
        for marker_root in dirs:
            path = Path(biopsy_path, marker_root, "results")
            print("Processing: ", marker_root)
            marker = marker_root
            for root, marker_sub_directories, files in os.walk(str(path)):
                if "experiment_run" in marker_sub_directories:
                    for experiment in marker_sub_directories:
                        model = LudwigModel.load(Path(path, experiment, 'model'))
                        eval_stats, _, _ = model.evaluate(dataset=dataset)

                        # Marker,MAE,MSE,RMSE,Biopsy,Panel,Type,Segmentation,SNR,FE,Mode,Hyper
                        print(eval_stats)

                        scores.append(
                            {
                                "Marker": marker,
                                "MAE": eval_stats[marker]['mean_absolute_error'],
                                "MSE": eval_stats[marker]['mean_squared_error'],
                                "RMSE": eval_stats[marker]['root_mean_squared_error'],
                                "Biopsy": test_biopsy,
                                "Combination": combination,
                                "FE": fe,
                                "Mode": "Ludwig",
                                "Hyper": 0
                            }
                        )

    scores = pd.DataFrame(scores)
    print(scores)
