import pandas as pd
import os, sys, json, shutil
from pathlib import Path
from tqdm import tqdm

EN_PATHS = [
    Path("mesmer", "tumor_in_patient_en"),
    Path("mesmer", "tumor_exp_patient_en"),
]

MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
           'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    save_path = Path("data/cleaned_data/scores/en")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    biopsy_data = {
        "9_2_1": [],
        "9_2_2": [],
        "9_3_1": [],
        "9_3_2": [],
        "9_14_1": [],
        "9_14_2": [],
        "9_15_1": [],
        "9_15_2": [],
    }

    ip_scores = pd.DataFrame()
    exp_scores = pd.DataFrame()
    for folder_path in EN_PATHS:
        mode = "IP" if "in_patient" in str(folder_path) else "EXP"
        # print(folder_path)
        for root, dirs, _ in os.walk(folder_path):
            for directory in dirs:
                biopsy_path = Path(root, directory)
                if mode == "EXP":
                    biopsy = str(biopsy_path).split('/')[-1]
                else:
                    biopsy = str(biopsy_path).split('/')[-1]
                    if biopsy[-1] == '1':
                        biopsy = biopsy[:-1] + '2'
                    else:
                        biopsy = biopsy[:-1] + '1'

                biopsy_path = Path(biopsy_path, biopsy)

                for marker in MARKERS:
                    for _, marker_directories, _ in os.walk(biopsy_path):
                        for marker_directory in marker_directories:
                            marker_dir = Path(biopsy_path, marker_directory)
                            marker = str(marker_dir).split('/')[-1]
                            for _, experiment_runs, _ in os.walk(marker_dir):
                                for experiment_run in experiment_runs:
                                    experiment_run_dir = Path(marker_dir, experiment_run)
                                    experiment_id: int = int(str(experiment_run_dir).split('/')[-1].split('_')[-1])

                                    try:
                                        # load json file using json library
                                        with open(Path(experiment_run_dir, "evaluation.json")) as file:
                                            evaluation = json.load(file)

                                    except:
                                        continue

                                    biopsy_dfs = biopsy_data[biopsy]
                                    # retrieve experiment run from list of dicts
                                    for experiment_run in biopsy_dfs:
                                        if experiment_id in experiment_run:
                                            experiment_run[experiment_id].append(
                                                {
                                                    "Biopsy": evaluation["biopsy"],
                                                    "Mode": mode,
                                                    "Experiment": experiment_id,
                                                    "Marker": marker,
                                                    "MAE": evaluation["mean_absolute_error"],
                                                    "RMSE": evaluation["root_mean_squared_error"],
                                                    "Network": "EN",
                                                    "Hyper": 0,
                                                    "FE": 0
                                                }
                                            )
                                        else:
                                            experiment_run[experiment_id] = [
                                                {
                                                    "Biopsy": evaluation["biopsy"],
                                                    "Mode": mode,
                                                    "Experiment": experiment_id,
                                                    "Marker": marker,
                                                    "MAE": evaluation["mean_absolute_error"],
                                                    "RMSE": evaluation["root_mean_squared_error"],
                                                    "Network": "EN",
                                                    "Hyper": 0,
                                                    "FE": 0
                                                }
                                            ]

                                    else:
                                        biopsy_dfs.append({
                                            experiment_id: [
                                                {
                                                    "Biopsy": evaluation["biopsy"],
                                                    "Mode": mode,
                                                    "Experiment": experiment_id,
                                                    "Marker": marker,
                                                    "MAE": evaluation["mean_absolute_error"],
                                                    "RMSE": evaluation["root_mean_squared_error"],
                                                    "Network": "EN",
                                                    "Hyper": 0,
                                                    "FE": 0
                                                }
                                            ]
                                        })

        print(f"Merging dataframes for mode: {mode}")

        for experiment_run in tqdm(biopsy_dfs):
            for experiment_id, data in experiment_run.items():
                if mode == "IP":
                    # merge dataframes for each experiment run


                    ip_scores = pd.concat([ip_scores, pd.DataFrame(data)], ignore_index=True)
                else:
                    exp_scores = pd.concat([exp_scores, pd.DataFrame(data)], ignore_index=True)

        print("Biopsy data reset...")
        biopsy_data = {
            "9_2_1": [],
            "9_2_2": [],
            "9_3_1": [],
            "9_3_2": [],
            "9_14_1": [],
            "9_14_2": [],
            "9_15_1": [],
            "9_15_2": [],
        }


    print("Saving scores...")
    scores = pd.concat([ip_scores, exp_scores], ignore_index=True)
    scores.to_csv(Path(save_path, "scores.csv"), index=False)
