import pandas as pd
import os, sys, json, shutil
from pathlib import Path
from tqdm import tqdm
from typing import List

EN_PATHS = [
    Path("mesmer", "tumor_in_patient_en"),
    Path("mesmer", "tumor_exp_patient_en"),
]

MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

if __name__ == '__main__':
    save_path = Path("data/cleaned_data/scores/en")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    test_scores: List = []

    for folder_path in EN_PATHS:
        mode = "IP" if "in_patient" in str(folder_path) else "EXP"
        # print(folder_path)
        for directory in os.listdir(folder_path):
            if not Path(folder_path, directory).is_dir():
                continue
            biopsy_path = Path(folder_path, directory)
            if mode == "EXP":
                biopsy = str(biopsy_path).split('/')[-1]
            else:
                biopsy = str(biopsy_path).split('/')[-1]
                if biopsy[-1] == '1':
                    biopsy = biopsy[:-1] + '2'
                else:
                    biopsy = biopsy[:-1] + '1'

            print(f"Loading biosy: {biopsy}...")
            for marker in MARKERS:
                print(f"Loading marker {marker} for biopsy {biopsy}...")
                marker_dir = Path(biopsy_path, biopsy, marker)
                for experiment_run in os.listdir(marker_dir):
                    if not Path(marker_dir, experiment_run).is_dir():
                        continue
                    experiment_run_dir = Path(marker_dir, experiment_run)
                    try:
                        experiment_id: int = int(str(experiment_run_dir).split('/')[-1].split('_')[-1])
                    except BaseException as ex:
                        print(ex)
                        print(experiment_run_dir)
                    try:
                        # load json file using json library
                        with open(Path(experiment_run_dir, "evaluation.json")) as file:
                            evaluation = json.load(file)

                    except BaseException as ex:
                        print(ex)
                        print(experiment_run_dir)
                        print(experiment_id)
                        continue

                    test_scores.append({
                        "Biopsy": evaluation["biopsy"],
                        "Mode": mode,
                        "Experiment": experiment_id,
                        "Marker": marker,
                        "MAE": evaluation["mean_absolute_error"],
                        "RMSE": evaluation["root_mean_squared_error"],
                        "Network": "EN",
                        "Hyper": 0,
                        "FE": 0
                    })

    print("Saving scores...")
    # convert biopsy data into dataframes
    scores = pd.DataFrame().from_records(test_scores)
    scores.to_csv(Path(save_path, "scores.csv"), index=False)
