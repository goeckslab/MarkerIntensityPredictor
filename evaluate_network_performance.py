import argparse, os
import shutil
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', "-m", type=str, required=True, help="The model to evaluate", choices=["ae", "gnn"])
    args = parser.parse_args()

    model = args.model
    if model == "ae":
        load_path = Path("ae_imputation")
    elif model == "gnn":
        load_path = Path("gnn/results")
    else:
        raise ValueError("Model not supported")

    all_scores = []
    loaded_files = 0
    for root, sub_directories, files in os.walk(load_path):
        for sub_directory in sub_directories:
            current_path = Path(root, sub_directory)
            try:
                if 'experiment_run' not in str(current_path):
                    continue
                print(f"Loading path: {Path(root, sub_directory)}")
                scores = pd.read_csv(Path(root, sub_directory, "scores.csv"))
                scores["Load Path"] = str(Path(current_path))
                all_scores.append(scores)

                loaded_files += 1

            except:
                continue

    all_scores = pd.concat(all_scores)

    if model == 'ae':
        save_path = Path("data/scores/ae")
    else:
        save_path = Path("data/scores/gnn")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    all_scores.to_csv(Path(save_path, "scores.csv"), index=False)
    print(f"Loaded {loaded_files} files.")
