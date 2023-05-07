import argparse, os
import shutil
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", type=str, required=True, help="path to ae_imputation folder")
    args = parser.parse_args()

    all_scores = []
    for root, sub_directories, files in os.walk(args.path):
        for sub_directory in sub_directories:
            print(Path(root, sub_directory))
            current_path = Path(root, sub_directory)
            try:
                if 'experiment_run' not in str(current_path):
                    continue

                scores = pd.read_csv(Path(root, sub_directory, "scores.csv"))
                experiment = str(current_path).split("/")[6].split("_")[-1]
                radius = str(current_path).split("/")[5]
                biopsy = str(current_path).split("/")[4]
                noise = str(current_path).split("/")[3]
                replace_value = str(current_path).split("/")[2]
                combination = str(current_path).split("/")[1]

                scores["FE"] = int(radius)
                scores["Replace Value"] = replace_value
                scores["Noise"] = noise
                scores["Experiment"] = experiment
                scores["Mode"] = "AE"

                all_scores.append(scores)

            except:
                continue

    all_scores = pd.concat(all_scores)
    save_path = Path("data/scores/ae")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    all_scores.to_csv(Path(save_path, "scores.csv"), index=False)
